//! Reading-order stage: port of indra_ocr/order_post.py (16-dim line
//! features, insertion local search, XY-cut baselines, average-linkage
//! clustering, block-order decoding).
//!
//! Unicode predicates (display width, isdigit, isupper) come from generated
//! codepoint run tables (char_tables.rs, scripts/gen-char-tables.py) taken
//! from the product interpreter itself — bit-identical to
//! unicodedata.east_asian_width / str.isdigit / str.isupper.
//!
//! f32 numerics mirror the product exactly: f32 boxes, f32 medians (mean of
//! the two middle f32 values), f32 log/log1p via std (measured bit-exact
//! vs numpy: /tmp f32chk on 408k samples), f32 logaddexp via
//! `-m - ln_1p(exp(-|a|))` with `m = max(0, -a)` (measured bit-exact vs
//! np.logaddexp(0, -a) on 408k samples).

pub const N_FEATS: usize = 16;
pub const N_BLOCK_CLASSES: usize = 10;

/// Chandra block labels in the order model's class order (cls head).
pub const BLOCK_CLASSES: [&str; N_BLOCK_CLASSES] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];

/// Block classes whose lines read row by row (grid labels): Table,
/// Table-Of-Contents, Form.
pub const GRID_CLASSES: [usize; 3] = [5, 8, 9];

const SENTENCE_END: &[char] = &['.', '!', '?', ':', ';', '。', '！', '？', '：', '；', '．'];
const LIST_MARKERS: &[char] = &['•', '-', '–', '▪', '◦', '*', '·', '→'];

mod char_tables {
    include!("char_tables.rs");

    /// True when codepoint `cp` lies in any run of `runs` (sorted, disjoint).
    pub fn in_runs(cp: u32, runs: &[(u32, u32)]) -> bool {
        runs.binary_search_by(|r| {
            use std::cmp::Ordering;
            if cp < r.0 {
                Ordering::Greater
            } else if cp > r.1 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
    }
}

/// True when the codepoint is a digit by Python's str.isdigit
/// (Nd + No-with-Numeric_Type=Digit; excludes Roman numerals and fractions).
pub fn is_py_digit(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::DIGIT_RUNS)
}

/// True when the codepoint is uppercase by Python's str.isupper (Lu + Lt).
pub fn is_py_upper(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::UPPER_RUNS)
}

/// Length with East Asian wide/fullwidth characters counted twice
/// (unicodedata.east_asian_width in "WF").
pub fn display_len(t: &str) -> usize {
    t.chars()
        .map(|c| if char_tables::in_runs(c as u32, char_tables::EAW_WF_RUNS) { 2 } else { 1 })
        .sum()
}

/// Python round(): round half to even (input is non-negative here).
fn py_round(x: f64) -> i64 {
    if x.fract() == 0.5 {
        let k = x.trunc() as i64;
        return if k % 2 == 0 { k } else { k + 1 };
    }
    x.round() as i64
}

/// Median of f32 values, as float64 of the f32 result (numpy keeps the
/// array dtype: even n -> mean of the two middle f32 values in f32).
fn median_f32(v: &[f32]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2] as f64
    } else {
        ((s[n / 2 - 1] + s[n / 2]) / 2.0f32) as f64
    }
}

fn boxes_f32(boxes: &[[f64; 4]]) -> Vec<[f32; 4]> {
    boxes
        .iter()
        .map(|b| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32])
        .collect()
}

/// [N, 4] boxes + page size (+ per-line texts) -> [N, 16] f32 features:
/// 10 geometry + 6 text. Mirrors order_post.line_features.
pub fn line_features(boxes: &[[f64; 4]], wh: (f64, f64), texts: Option<&[String]>) -> Vec<f32> {
    let (pw, ph) = (wh.0 as f32, wh.1 as f32);
    let b = boxes_f32(boxes);
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }
    let w: Vec<f32> = b.iter().map(|x| (x[2] - x[0]).max(1e-3f32)).collect();
    let h: Vec<f32> = b.iter().map(|x| (x[3] - x[1]).max(1e-3f32)).collect();
    let med_h = median_f32(&h);
    // Product: np.full(n, med_h / H * 20.0) with med_h a Python float — f64
    // arithmetic, cast to f32 by the final .astype(np.float32).
    let col9 = (med_h / wh.1 * 20.0) as f32;

    let mut out = vec![0f32; n * N_FEATS];
    for i in 0..n {
        let bi = b[i];
        let row = &mut out[i * N_FEATS..(i + 1) * N_FEATS];
        row[0] = bi[0] / pw;
        row[1] = bi[1] / ph;
        row[2] = bi[2] / pw;
        row[3] = bi[3] / ph;
        row[4] = (bi[0] + bi[2]) / (2.0 * pw);
        row[5] = (bi[1] + bi[3]) / (2.0 * ph);
        row[6] = w[i] / pw;
        row[7] = (h[i] / med_h as f32).ln();
        row[8] = (w[i] / h[i]).ln() / 4.0;
        row[9] = col9;

        if let Some(texts) = texts {
            let t = texts[i].trim();
            let chars: Vec<char> = t.chars().collect();
            if !chars.is_empty() {
                row[10] = ((display_len(t) as f64).ln_1p() / 5.0) as f32;
                let nd = chars.iter().filter(|c| is_py_digit(**c)).count() as f64;
                row[11] = (nd / chars.len() as f64) as f32;
                // t[0] in LIST_MARKERS, or a leading enumerator ("1.", "(2)"):
                // t[:3].rstrip(".)").isdigit() and len(t) > 3
                let first3: String = chars.iter().take(3).collect();
                let stripped = first3.trim_end_matches(['.', ')']);
                let enumerator = !stripped.is_empty()
                    && stripped.chars().all(is_py_digit)
                    && chars.len() > 3;
                row[12] = (LIST_MARKERS.contains(&chars[0]) || enumerator) as u8 as f32;
                row[13] = (chars.last() == Some(&'-')) as u8 as f32;
                row[14] = SENTENCE_END.contains(&chars[chars.len() - 1]) as u8 as f32;
                row[15] = is_py_upper(chars[0]) as u8 as f32;
            }
        }
    }
    out
}

/// Reading order from a pairwise logit matrix [N, N] (a[i][j] > 0: i before
/// j): row-sum ranking, then `sweeps` of insertion local search. Mirrors
/// order_post.decode_order.
pub fn decode_order(a: &[f32], n: usize, sweeps: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }
    // lp[i][j] = log P(i before j) = -logaddexp(0, -a[i][j]), f32.
    let lp: Vec<f32> = a
        .iter()
        .map(|&av| {
            let m = (-av).max(0.0);
            -m - (-av.abs()).exp().ln_1p()
        })
        .collect();
    // score = -rowsum(lp) + colsum(lp), f32 elementwise (product order).
    let mut row = vec![0f32; n];
    for i in 0..n {
        let mut s = 0f32;
        for j in 0..n {
            s += lp[i * n + j];
        }
        row[i] = s;
    }
    let mut col = vec![0f32; n];
    for j in 0..n {
        for i in 0..n {
            col[i] += lp[i * n + j];
        }
    }
    let mut score = vec![0f32; n];
    for i in 0..n {
        score[i] = -row[i] + col[i];
    }
    // np.argsort (ascending, stable): (score, index)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        score[i]
            .total_cmp(&score[j])
            .then_with(|| i.cmp(&j))
    });

    let mut improved = true;
    let mut sweep = 0usize;
    while improved && sweep < sweeps {
        improved = false;
        let snapshot = order.clone();
        for &x in &snapshot {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            let m = n - 1;
            let mut cb = vec![0f32; m]; // cumsum(before[k])
            let mut ra = vec![0f32; m]; // sum(after[k..])
            for k in 0..m {
                cb[k] = lp[rest[k] * n + x]; // x placed after rest[k]
                ra[k] = lp[x * n + rest[k]]; // x placed before rest[k]
            }
            if m > 1 {
                for k in 1..m {
                    cb[k] += cb[k - 1];
                }
                for k in (0..m - 1).rev() {
                    ra[k] += ra[k + 1];
                }
            }
            // gains[k] = (k==0 ? 0 : cb[k-1]) + (k==m ? 0 : ra[k])
            let mut gains = vec![0f32; n];
            gains[0] = ra[0]; // inserting first: x before every line of rest
            for k in 1..n {
                let b = if k == m { 0.0f32 } else { ra[k] };
                gains[k] = cb[k - 1] + b;
            }
            let mut k = 0usize;
            for (i, &g) in gains.iter().enumerate() {
                if g > gains[k] {
                    k = i;
                }
            }
            let cur = order.iter().position(|&v| v == x).unwrap();
            if gains[k] > gains[cur] + 1e-6 {
                let mut next = rest;
                next.insert(k, x);
                order = next;
                improved = true;
            }
        }
        sweep += 1;
    }
    order
}

/// Rows top to bottom (centres within half a line height share a row),
/// left to right. Mirrors order_post.order_top_down on f32 boxes.
pub fn order_top_down(boxes: &[[f32; 4]]) -> Vec<usize> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }
    let cy: Vec<f32> = (0..n).map(|i| (boxes[i][1] + boxes[i][3]) / 2.0f32).collect();
    let h = median_f32(&(0..n).map(|i| boxes[i][3] - boxes[i][1]).collect::<Vec<_>>()) as f32;
    let row_of = |i: usize| py_round((cy[i] / (h * 0.5f32).max(1e-3f32)) as f64);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        let (ri, rj) = (row_of(i), row_of(j));
        ri.cmp(&rj).then_with(|| (boxes[i][0] as f64).total_cmp(&(boxes[j][0] as f64)))
    });
    idx
}

fn best_cut(b: &[[f32; 4]], axis: usize) -> (f32, i32, Vec<usize>) {
    let (lo_i, hi_i) = if axis == 1 { (1, 3) } else { (0, 2) };
    let n = b.len();
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|&i, &j| b[i][lo_i].total_cmp(&b[j][lo_i]).then(i.cmp(&j))); // np.argsort stable
    let mut end = b[o[0]][hi_i];
    let mut best_gap = 0.0f32;
    let mut best_k = -1i32;
    for k in 1..n {
        let gap = b[o[k]][lo_i] - end;
        if gap > best_gap {
            best_gap = gap;
            best_k = k as i32;
        }
        if b[o[k]][hi_i] > end {
            end = b[o[k]][hi_i];
        }
    }
    (best_gap, best_k, o)
}

/// Recursive XY-cut: split at the widest whitespace gap on either axis
/// (threshold 0); a horizontal cut puts the upper part first, a vertical cut
/// the left part first; without a gap, fall back to top-down order. Mirrors
/// order_post.order_xycut.
pub fn order_xycut(boxes: &[[f32; 4]]) -> Vec<usize> {
    fn rec(boxes: &[[f32; 4]], ids: &[usize]) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids.to_vec();
        }
        let b: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
        let (gy, ky, oy) = best_cut(&b, 1);
        let (gx, kx, ox) = best_cut(&b, 0);
        if ky < 0 && kx < 0 {
            return order_top_down(&b)
                .iter()
                .map(|&i| ids[i])
                .collect();
        }
        let (k, o): (i32, Vec<usize>) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let left: Vec<usize> = o[..k as usize].iter().map(|&i| ids[i]).collect();
        let right: Vec<usize> = o[k as usize..].iter().map(|&i| ids[i]).collect();
        rec(boxes, &left)
            .into_iter()
            .chain(rec(boxes, &right))
            .collect()
    }
    rec(boxes, &(0..boxes.len()).collect::<Vec<_>>())
}

/// Blocks in reading order by recursive XY-cut with a gap threshold of
/// `min_gap` median line heights; lines inside a block go by order_xycut.
/// Mirrors order_post.xycut_blocks.
pub fn xycut_blocks(boxes: &[[f32; 4]], min_gap: f64) -> Vec<Vec<usize>> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let h = median_f32(&(0..boxes.len()).map(|i| boxes[i][3] - boxes[i][1]).collect::<Vec<_>>());
    let ids: Vec<usize> = (0..boxes.len()).collect();
    rec_blocks(boxes, min_gap, h, &ids)
}

fn rec_blocks(boxes: &[[f32; 4]], min_gap: f64, h: f64, ids: &[usize]) -> Vec<Vec<usize>> {
    if ids.len() > 1 {
        let b: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
        let (gy, ky, oy) = best_cut(&b, 1);
        let (gx, kx, ox) = best_cut(&b, 0);
        let (gap, k, o): (f32, i32, Vec<usize>) =
            if gy >= gx { (gy, ky, oy) } else { (gx, kx, ox) };
        if k >= 0 && gap as f64 >= min_gap * h {
            let left_ids: Vec<usize> = o[..k as usize].iter().map(|&i| ids[i]).collect();
            let right_ids: Vec<usize> = o[k as usize..].iter().map(|&i| ids[i]).collect();
            let left = rec_blocks(boxes, min_gap, h, &left_ids);
            let right = rec_blocks(boxes, min_gap, h, &right_ids);
            return left.into_iter().chain(right).collect();
        }
    }
    let b: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
    let local = order_xycut(&b);
    vec![local.iter().map(|&i| ids[i]).collect()]
}

/// Average-linkage clustering of pairwise same-group logits: merge the two
/// groups with the highest mean logit while that mean is positive. Mirrors
/// order_post._cluster (incremental best-partner maintenance; verified
/// equivalent to the naive O(n^3) merge on 300 random matrices).
pub fn cluster(same: &[f32], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let mut s = vec![vec![0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            s[i][j] = if i == j { 0.0 } else { same[i * n + j] as f64 };
        }
    }
    let mut size = vec![1.0f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m: Vec<Vec<f64>> = s.clone();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = f64::NEG_INFINITY;
    }
    let argmax_row = |row: &[f64]| -> usize {
        let mut k = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[k] {
                k = i;
            }
        }
        k
    };
    let mut best = vec![0usize; n];
    let mut best_v = vec![0f64; n];
    for i in 0..n {
        best[i] = argmax_row(&m[i]);
        best_v[i] = m[i][best[i]];
    }
    for _ in 0..n - 1 {
        let mut a = 0usize;
        for (i, &v) in best_v.iter().enumerate() {
            if v > best_v[a] {
                a = i;
            }
        }
        if best_v[a] <= 0.0 {
            break;
        }
        let b = best[a];
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        // column a += column b (indexed: a and b are disjoint rows of s)
        #[allow(clippy::needless_range_loop)]
        {
            for j in 0..n {
                s[a][j] += s[b][j];
            }
        }
        for row in s.iter_mut() {
            row[a] += row[b];
        }
        s[a][a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let mb = std::mem::take(&mut members[b]);
        members[a].extend(mb);
        let mut row = vec![0f64; n];
        for j in 0..n {
            row[j] = s[a][j] / (size[a] * size[j]);
            if !alive[j] {
                row[j] = f64::NEG_INFINITY;
            }
        }
        row[a] = f64::NEG_INFINITY;
        for j in 0..n {
            m[a][j] = row[j];
            m[j][a] = row[j];
            m[b][j] = f64::NEG_INFINITY;
            m[j][b] = f64::NEG_INFINITY;
        }
        best_v[b] = f64::NEG_INFINITY;
        best[a] = argmax_row(&row);
        best_v[a] = row[best[a]];
        // other rows: (j, a) changed and (j, b) is gone
        for j in 0..n {
            if j != a && alive[j] && (best[j] == a || best[j] == b) {
                best[j] = argmax_row(&m[j]);
                best_v[j] = m[j][best[j]];
            }
        }
        for j in 0..n {
            if j != a && alive[j] && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..n {
        if alive[i] {
            let mut mem = members[i].clone();
            mem.sort_unstable();
            out.push(mem);
        }
    }
    out
}

/// Reading order with the layout heads: blocks from average-linkage
/// clustering of the same-block logits, lines inside a block by XY-cut (or
/// row by row in a grid-class block), blocks by the mean pairwise order
/// logit between them. Mirrors order_post.decode_blocks_order.
pub fn decode_blocks_order(
    a: &[f32],
    same: &[f32],
    boxes: &[[f64; 4]],
    cls: Option<&[f32]>,
) -> Vec<usize> {
    let n = (a.len() as f64).sqrt() as usize;
    debug_assert_eq!(a.len(), n * n);
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(same, n);
    let f32b = boxes_f32(boxes);
    let mut inner: Vec<Vec<usize>> = Vec::new();
    for b in &blocks {
        let bx: Vec<[f32; 4]> = b.iter().map(|&i| f32b[i]).collect();
        let mut grid = false;
        if let Some(cls) = cls {
            let mut colsum = [0f32; N_BLOCK_CLASSES];
            for &i in b {
                for c in 0..N_BLOCK_CLASSES {
                    colsum[c] += cls[i * N_BLOCK_CLASSES + c];
                }
            }
            let mut am = 0usize;
            for (c, &v) in colsum.iter().enumerate() {
                if v > colsum[am] {
                    am = c;
                }
            }
            grid = GRID_CLASSES.contains(&am);
        }
        let local = if grid {
            order_top_down(&bx)
        } else {
            order_xycut(&bx)
        };
        // local[k] is the position of the k-th line within the block (product:
        // inner.append([b[k] for k in local])).
        inner.push(local.iter().map(|&k| b[k]).collect());
    }
    // Product: up = triu(G a G.T, 1), ab = up - up.T. M[k][l] = mean of
    // a[i][j] over i in block k, j in block l (G[k][i] = 1/|block|). Only the
    // strictly upper triangle of M is kept; the lower triangle of ab is its
    // sign flip (the antisymmetric "k before l" block logit).
    let m = blocks.len();
    let mut up = vec![vec![0f64; m]; m];
    for k in 0..m {
        for l in (k + 1)..m {
            let mut sum = 0f64;
            for &i in &blocks[k] {
                for &j in &blocks[l] {
                    sum += a[i * n + j] as f64;
                }
            }
            up[k][l] = sum / (blocks[k].len() as f64 * blocks[l].len() as f64);
        }
    }
    let mut ab = vec![0f32; m * m];
    for k in 0..m {
        for l in 0..m {
            if k == l {
                continue;
            }
            ab[k * m + l] = (up[k][l] - up[l][k]) as f32;
        }
    }
    let bo = decode_order(&ab, m, 4);
    bo.iter().flat_map(|&k| inner[k].clone()).collect()
}

/// Reading order of a page: the product's pipeline.reading_order. Boxes with
/// fewer than two lines (or no model) fall back to XY-cut.
pub struct ReadingOrder {
    pub order: Vec<usize>,
    /// Pairwise order logits [N, N] f32 (model head), None on the xycut path.
    pub a: Option<Vec<f32>>,
    /// Same-block logits [N, N] f32.
    pub same: Option<Vec<f32>>,
    /// Per-line block-class logits [N, 10] f32.
    pub cls: Option<Vec<f32>>,
    /// Average-linkage blocks (line-index groups), None on the xycut path.
    pub blocks: Option<Vec<Vec<usize>>>,
}

impl super::IndraEngine {
    /// Reading order for det boxes + recognized texts (product
    /// `reading_order`): feats -> order.onnx (feats [1,N,16] + all-true mask)
    /// -> decode_blocks_order on the a/same/cls heads.
    pub fn reading_order(
        &self,
        boxes: &[[f64; 4]],
        texts: &[String],
        wh: (f64, f64),
    ) -> Result<ReadingOrder, String> {
        let n = boxes.len();
        if n < 2 {
            let f32b = boxes_f32(boxes);
            let order = if n == 0 { Vec::new() } else { order_xycut(&f32b) };
            return Ok(ReadingOrder {
                order,
                a: None,
                same: None,
                cls: None,
                blocks: None,
            });
        }
        let feats = line_features(boxes, wh, Some(texts));
        let mask = vec![1i32; n]; // product: np.ones((1, N), bool)
        let outs = self
            .order
            .run_fi("feats", &[1, n, N_FEATS], &feats, "mask", &[1, n], &mask)?;
        // Model output names (verified on order.onnx): a, cls, same, tab.
        let by_name = |outs: &[super::Tensor], name: &str| -> Result<Vec<f32>, String> {
            for (i, (on, _)) in self.order.outputs.iter().enumerate() {
                if on == name {
                    return outs[i]
                        .as_float()
                        .cloned()
                        .ok_or_else(|| format!("order head '{name}' is not float"));
                }
            }
            Err(format!("order head '{name}' not in outputs"))
        };
        let a = by_name(&outs, "a")?;
        let cls = by_name(&outs, "cls")?;
        let same = by_name(&outs, "same")?;
        if a.len() != n * n || cls.len() != n * N_BLOCK_CLASSES || same.len() != n * n {
            return Err(format!(
                "order head shapes a={:?} cls={:?} same={:?} for N={n}",
                a.len(),
                cls.len(),
                same.len()
            ));
        }
        let blocks = cluster(&same, n);
        let order = decode_blocks_order(&a, &same, boxes, Some(&cls));
        Ok(ReadingOrder {
            order,
            a: Some(a),
            same: Some(same),
            cls: Some(cls),
            blocks: Some(blocks),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_len_counts_wide_double() {
        assert_eq!(display_len("hello"), 5);
        assert_eq!(display_len("中文"), 4);
        assert_eq!(display_len("a中b"), 4);
        assert_eq!(display_len("Ｈｅｌｌｏ"), 10);
        assert_eq!(display_len("Ⅰ½İ½"), 4); // narrow even though exotic
        assert_eq!(display_len(""), 0);
    }

    #[test]
    fn py_char_predicates() {
        assert!(is_py_digit('5'));
        assert!(is_py_digit('٥')); // Arabic-Indic
        assert!(!is_py_digit('Ⅻ')); // Roman numeral: No, not a Digit
        assert!(!is_py_digit('½')); // fraction: No
        assert!(is_py_upper('A'));
        assert!(is_py_upper('İ')); // Lt
        assert!(is_py_upper('Ⅰ')); // Roman numerals are Lt
        assert!(!is_py_upper('a'));
        assert!(!is_py_upper('ª'));
    }

    #[test]
    fn decode_order_trivial_and_clear() {
        assert_eq!(decode_order(&[] as &[f32], 0, 4), Vec::<usize>::new());
        assert_eq!(decode_order(&[0.0], 1, 4), vec![0]);
        // Strong 0-before-1 signal.
        let a = [0.0f32, 3.0, -3.0, 0.0];
        assert_eq!(decode_order(&a, 2, 4), vec![0, 1]);
        // Reversed.
        let a = [0.0f32, -3.0, 3.0, 0.0];
        assert_eq!(decode_order(&a, 2, 4), vec![1, 0]);
    }

    #[test]
    fn cluster_splinks_positive_means() {
        // Two tight groups, negative between them.
        let same = [
            0.0f32, 2.0, -1.0,
            2.0, 0.0, -1.0,
            -1.0, -1.0, 0.0,
        ];
        let c = cluster(&same, 3);
        assert_eq!(c, vec![vec![0, 1], vec![2]]);
        // All negative means: every line its own group.
        let same = [0.0f32, -0.5, -0.5, -0.5, 0.0, -0.5, -0.5, -0.5, 0.0];
        let c = cluster(&same, 3);
        assert_eq!(c, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(cluster(&[], 0), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn xycut_two_columns() {
        // Two vertical columns separated by a wide gap.
        let boxes = [
            [10.0, 10.0, 90.0, 30.0],
            [10.0, 40.0, 90.0, 60.0],
            [200.0, 10.0, 280.0, 30.0],
            [200.0, 40.0, 280.0, 60.0],
        ];
        let order = order_xycut(&boxes);
        assert_eq!(order, vec![0, 1, 2, 3]);
        let blocks = xycut_blocks(&boxes, 1.0);
        assert_eq!(blocks, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn top_down_rows() {
        // Two rows, two lines each, interleaved x.
        let boxes = [
            [50.0, 10.0, 150.0, 30.0],
            [10.0, 12.0, 90.0, 32.0],
            [10.0, 50.0, 90.0, 70.0],
            [50.0, 52.0, 150.0, 72.0],
        ];
        let order = order_top_down(&boxes);
        assert_eq!(order, vec![1, 0, 2, 3]);
    }

    #[test]
    fn line_features_shape_and_text_flags() {
        let boxes = [[10.0, 20.0, 110.0, 40.0], [10.0, 60.0, 110.0, 80.0]];
        let texts = vec!["Hello world.".to_string(), "１２３".to_string()];
        let f = line_features(&boxes, (400.0, 400.0), Some(&texts));
        assert_eq!(f.len(), 2 * N_FEATS);
        // line 0: ends with '.', starts upper
        assert_eq!(f[14], 1.0f32);
        assert_eq!(f[15], 1.0f32);
        assert_eq!(f[12], 0.0);
        assert_eq!(f[13], 0.0);
        // line 1: CJK fullwidth digits: digit fraction 1.0, not upper
        assert_eq!(f[N_FEATS + 11], 1.0f32);
        assert_eq!(f[N_FEATS + 15], 0.0);
        assert_eq!(f[N_FEATS + 14], 0.0);
        // empty text: all text features 0
        let f2 = line_features(&boxes, (400.0, 400.0), Some(&["".to_string(), "  ".to_string()]));
        for i in 10..16 {
            assert_eq!(f2[i], 0.0);
            assert_eq!(f2[N_FEATS + i], 0.0);
        }
        assert_eq!(line_features(&[], (10.0, 10.0), None), Vec::<f32>::new());
    }
}
