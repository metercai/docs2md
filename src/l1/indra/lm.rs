//! P0: the n-gram language model (port of indra_ocr/lm.py, NgramLM).
//!
//! A key packs tokens `bits` bits each (0 = line start, t+1 = token t); the
//! history length goes in the top 3 bits (TAG_SHIFT = 61), so a search over
//! one concatenated per-length table answers every history length at once.
//!
//! Python reference (deterministic, no caches needed for correctness):
//! logp = value of the longest seen history n <= m that saw the token, plus
//! the backoff weights of the longer seen histories (n, m]; no such n ->
//! all backoffs + uniform floor -log(vocab_size).

use std::collections::HashMap;
use std::sync::OnceLock;

const TAG_SHIFT: u32 = 61;
const BOS: i64 = -1;
const BITS_DEFAULT: u32 = 11;

pub(crate) fn pack(tokens: &[i64], bits: u32) -> u64 {
    let mut k: u64 = 0;
    for t in tokens {
        let v = if *t == BOS { 0 } else { (*t + 1) as u64 };
        k = (k << bits) | v;
    }
    k
}

/// One parsed LM table set, per history length n (0..order) — mirroring the
/// Python per-length tables exactly (a merged tagged table cannot represent
/// "not present in length n" for n=0, whose tag is 0).
pub struct NgramLm {
    pub order: u32,
    pub vocab_size: i64,
    pub bits: u32,
    /// p_keys[n]/p_vals[n]: (history, token) -> log-prob for histories of
    /// length n, keys packed with the token in the low bits, sorted.
    pub(crate) p_keys: Vec<Vec<u64>>,
    pub(crate) p_vals: Vec<Vec<f64>>,
    /// h_keys[n]/h_vals[n]: history -> backoff weight for length n, sorted.
    pub(crate) h_keys: Vec<Vec<u64>>,
    pub(crate) h_vals: Vec<Vec<f64>>,
    /// Per-length backoff arrays (for max_logp).
    hb: Vec<Vec<f32>>,
    max_logp: OnceLock<f64>,
    cache: HashMap<(Vec<i64>, i64), f64>,
}

impl NgramLm {
    /// Parse from parsed npz arrays (keys "order", "vocab_size", "bits",
    /// "k{n}"/"v{n}"/"hk{n}"/"hb{n}").
    pub fn from_arrays(arr: &HashMap<String, super::npz::NpyArray>) -> Result<Self, String> {
        use super::npz::NpyArray;
        let scalar = |k: &str| -> Result<i64, String> {
            match arr.get(k) {
                Some(NpyArray::ScalarI64(v)) => Ok(*v),
                _ => Err(format!("lm.npz: missing scalar {k}")),
            }
        };
        let order = scalar("order")? as u32;
        let vocab_size = scalar("vocab_size")?;
        let bits = match arr.get("bits") {
            Some(NpyArray::ScalarI64(b)) => *b as u32,
            _ => BITS_DEFAULT,
        };
        if order as u64 * bits as u64 > TAG_SHIFT as u64 {
            return Err(format!(
                "order {order} x {bits} bits do not fit a key"
            ));
        }
        let keys_of = |k: &str| -> Result<Vec<u64>, String> {
            match arr.get(k) {
                Some(NpyArray::Keys(v)) => Ok(v.clone()),
                _ => Err(format!("lm.npz: missing keys {k}")),
            }
        };
        let vals_of = |k: &str| -> Result<Vec<f32>, String> {
            match arr.get(k) {
                Some(NpyArray::F32(v)) => Ok(v.clone()),
                _ => Err(format!("lm.npz: missing values {k}")),
            }
        };
        let mut p_keys: Vec<Vec<u64>> = Vec::new();
        let mut p_vals: Vec<Vec<f64>> = Vec::new();
        let mut h_keys: Vec<Vec<u64>> = Vec::new();
        let mut h_vals: Vec<Vec<f64>> = Vec::new();
        let mut backoffs: Vec<Vec<f32>> = Vec::new();
        for n in 0..order {
            let k = format!("k{n}");
            let v = format!("v{n}");
            let hk = format!("hk{n}");
            let hbk = format!("hb{n}");
            let (k, v, hk, hbk) = (keys_of(&k)?, vals_of(&v)?, keys_of(&hk)?, vals_of(&hbk)?);
            p_keys.push(k);
            p_vals.push(v.iter().map(|x| *x as f64).collect());
            h_keys.push(hk);
            h_vals.push(hbk.iter().map(|x| *x as f64).collect());
            backoffs.push(hbk);
        }
        for (k, v) in p_keys.iter_mut().zip(p_vals.iter_mut()) {
            sort_pairs(k, v);
        }
        for (k, v) in h_keys.iter_mut().zip(h_vals.iter_mut()) {
            sort_pairs(k, v);
        }
        Ok(Self {
            order,
            vocab_size,
            bits,
            p_keys,
            p_vals,
            h_keys,
            h_vals,
            hb: backoffs,
            max_logp: OnceLock::new(),
            cache: HashMap::new(),
        })
    }

    /// Upper bound of every logp result (Python `max_logp`).
    pub fn max_logp(&self) -> f64 {
        *self.max_logp.get_or_init(|| {
            let top = self
                .p_vals
                .iter()
                .flatten()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .max((-self.vocab_size as f64).ln());
            let bo: f64 = self
                .hb
                .iter()
                .filter(|h| !h.is_empty())
                .map(|h| {
                    h.iter()
                        .cloned()
                        .map(f64::from)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .map(|m| m.max(0.0))
                .sum();
            top + bo
        })
    }

    fn find(keys: &[u64], key: u64) -> isize {
        match keys.binary_search(&key) {
            Ok(i) => i as isize,
            Err(_) => -1,
        }
    }

    /// logp(hist, t) — exact port of the Python backoff loop, per-length tables.
    pub fn logp(&mut self, hist: &[i64], t: i64) -> f64 {
        let ck = (hist.to_vec(), t);
        if let Some(v) = self.cache.get(&ck) {
            return *v;
        }
        let top = (hist.len() as u32).min(self.order - 1) as usize;
        // hs[n] = last n tokens (n = 0..=top)
        let mut hs: Vec<Vec<i64>> = Vec::with_capacity(top + 1);
        for n in 0..=top {
            if n == 0 {
                hs.push(Vec::new());
            } else {
                hs.push(hist[hist.len() - n..].to_vec());
            }
        }
        let mut m = 0usize;
        for (n, h) in hs.iter().enumerate().skip(1) {
            if Self::find(&self.h_keys[n], pack(h, self.bits)) < 0 {
                break;
            }
            m = n;
        }
        let (mut acc, mut v) = (0.0f64, None);
        for n in (0..=m).rev() {
            let pkey = pack_key_with_token(&hs[n], t, self.bits);
            if let Ok(i) = self.p_keys[n].binary_search(&pkey) {
                v = Some(acc + self.p_vals[n][i]);
                break;
            }
            let hkey = pack(&hs[n], self.bits);
            let j = Self::find(&self.h_keys[n], hkey);
            acc += self.h_vals[n][j as usize];
        }
        let v = v.unwrap_or_else(|| acc - (self.vocab_size as f64).ln());
        if self.cache.len() < 1_000_000 {
            self.cache.insert(ck, v);
        }
        v
    }
}

/// Pack history then the token (token in the low bits): (k << bits) | t+1.
fn pack_key_with_token(hist: &[i64], t: i64, bits: u32) -> u64 {
    let mut k: u64 = 0;
    for x in hist {
        k = (k << bits) | (if *x == BOS { 0 } else { (*x + 1) as u64 });
    }
    (k << bits) | (if t == BOS { 0 } else { (t + 1) as u64 })
}

fn sort_pairs(keys: &mut Vec<u64>, vals: &mut Vec<f64>) {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    let nk = idx.iter().map(|&i| keys[i]).collect();
    let nv = idx.iter().map(|&i| vals[i]).collect();
    *keys = nk;
    *vals = nv;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from the Python NgramLM on the embedded lm.npz
    // (generated with indra-ocr 1.1.2, /tmp/indra-p0-ref.json).
    const LM: &[u8] = include_bytes!("../../../assets/models/indra/lm.npz");
    const LM_CJK: &[u8] = include_bytes!("../../../assets/models/indra/lm_cjk.npz");
    const LM_WORLD: &[u8] = include_bytes!("../../../assets/models/indra/lm_world.npz");

    fn load(bytes: &[u8]) -> NgramLm {
        let arr = super::super::npz::parse_npz(bytes).expect("parse npz");
        NgramLm::from_arrays(&arr).expect("lm")
    }

    #[test]
    fn parse_all_three() {
        let b = load(LM);
        assert_eq!((b.order, b.vocab_size, b.bits), (5, 1678, 11));
        let c = load(LM_CJK);
        assert_eq!((c.order, c.vocab_size, c.bits), (3, 11750, 14));
        let w = load(LM_WORLD);
        assert_eq!((w.order, w.vocab_size, w.bits), (4, 2936, 12));
    }

    #[test]
    fn logp_matches_python() {
        // Reference values from the Python NgramLM on the embedded npz files
        // (indra-ocr 1.1.2, regenerated 2026-11-04: /tmp/indra-p0-logp.json).
        let base = [
            (vec![], 1, -1.939063310623169f64),
            (vec![], 500, -18.328623840091115),
            (vec![5], 7, -8.470968246459961),
            (vec![12, 3], 9, -7.943770885467529),
            (vec![3, 1, 2], 4, -9.039415836334229),
            (vec![99, 8, 7], 2, -9.83804702758789),
        ];
        let mut lm = load(LM);
        for (h, t, refp) in base {
            let got = lm.logp(&h, t);
            assert!(
                (got - refp).abs() < 1e-6,
                "base hist={h:?} t={t}: got {got} want {refp}"
            );
        }
        let cjk = [
            (vec![], 1, -2.5946991443634033f64),
            (vec![], 500, -17.65841176511674),
            (vec![5], 7, -12.35056734085083),
            (vec![12, 3], 9, -4.642394065856934),
            (vec![3, 1, 2], 4, -14.000889301300049),
            (vec![99, 8, 7], 2, -12.951240301132202),
        ];
        let mut lm = load(LM_CJK);
        for (h, t, refp) in cjk {
            let got = lm.logp(&h, t);
            assert!(
                (got - refp).abs() < 1e-6,
                "cjk hist={h:?} t={t}: got {got} want {refp}"
            );
        }
        let world = [
            (vec![], 1, -2.495425224304199f64),
            (vec![], 500, -18.197585295862336),
            (vec![5], 7, -13.438922643661499),
            (vec![12, 3], 9, -5.892481327056885),
            (vec![3, 1, 2], 4, -17.764718532562256),
            (vec![99, 8, 7], 2, -12.292409658432007),
        ];
        let mut lm = load(LM_WORLD);
        for (h, t, refp) in world {
            let got = lm.logp(&h, t);
            assert!(
                (got - refp).abs() < 1e-6,
                "world hist={h:?} t={t}: got {got} want {refp}"
            );
        }
    }

    #[test]
    fn max_logp_sane() {
        let lm = load(LM);
        let m = lm.max_logp();
        assert!(m > -2.0 && m < 0.0, "max_logp {m}");
    }
}
