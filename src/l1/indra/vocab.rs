//! P0: the script-group vocabularies from meta.json.
//!
//! meta.json holds, per group: a JSON string `{"blank": 0, "chars": [..]}`
//! where class i decodes to `chars[i-1]` (class 0 is blank). The rec models
//! are per-group (rec.onnx=base, rec_cjk, rec_rtl, rec_world); script.onnx
//! routes a line crop to its group (P3).

use std::collections::HashMap;

pub const GROUPS: [&str; 4] = ["base", "cjk", "rtl", "world"];

#[derive(Debug, Clone)]
pub struct Vocab {
    pub chars: Vec<String>,
}

impl Vocab {
    /// class -> char (class 0 = blank is out of range; caller skips it).
    pub fn decode(&self, class: i32) -> Option<&str> {
        if class <= 0 {
            return None;
        }
        self.chars.get((class - 1) as usize).map(|s| s.as_str())
    }
    pub fn size(&self) -> usize {
        self.chars.len() + 1 // + blank
    }
}

#[derive(Debug, Clone)]
pub struct Meta {
    pub det_long_side: u32,
    pub rec_height: u32,
    pub rec_window: u32,
    pub script_groups: Vec<String>,
    pub lm_params: HashMap<String, LmParams>,
    pub vocab: HashMap<String, Vocab>,
}

#[derive(Debug, Clone, Copy)]
pub struct LmParams {
    pub alpha: f64,
    pub beta: f64,
    pub gate: f64,
}

impl Meta {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let m: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("meta.json: {e}"))?;
        let num = |k: &str| -> Result<u32, String> {
            m.get(k)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("meta.json: missing {k}"))
                .map(|v| v as u32)
        };
        let mut vocab = HashMap::new();
        for g in GROUPS {
            let key = if g == "base" { "vocab" } else { &format!("vocab_{g}") };
            let s = m
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("meta.json: missing {key}"))?;
            let v: serde_json::Value =
                serde_json::from_str(s).map_err(|e| format!("{key}: {e}"))?;
            let chars: Vec<String> = v
                .get("chars")
                .and_then(|c| c.as_array())
                .ok_or_else(|| format!("{key}: missing chars"))?
                .iter()
                .map(|c| c.as_str().unwrap_or_default().to_string())
                .collect();
            vocab.insert(g.to_string(), Vocab { chars });
        }
        let mut lm_params = HashMap::new();
        if let Some(p) = m.get("lm_params").and_then(|v| v.as_object()) {
            for (k, v) in p {
                lm_params.insert(
                    k.clone(),
                    LmParams {
                        alpha: v.get("alpha").and_then(|x| x.as_f64()).unwrap_or(0.2),
                        beta: v.get("beta").and_then(|x| x.as_f64()).unwrap_or(1.5),
                        gate: v.get("gate").and_then(|x| x.as_f64()).unwrap_or(0.95),
                    },
                );
            }
        }
        Ok(Self {
            det_long_side: num("det_long_side")?,
            rec_height: num("rec_height")?,
            rec_window: num("rec_window")?,
            script_groups: m
                .get("script_groups")
                .and_then(|v| v.as_array())
                .ok_or("meta.json: missing script_groups")?
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            lm_params,
            vocab,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: &[u8] = include_bytes!("../../../assets/models/indra/meta.json");

    #[test]
    fn meta_parses() {
        let m = Meta::parse(META).expect("meta");
        assert_eq!(m.det_long_side, 1024);
        assert_eq!(m.rec_height, 32);
        assert_eq!(m.rec_window, 3);
        assert_eq!(m.script_groups, vec!["base", "cjk", "rtl", "world"]);
        assert!(m.vocab["base"].decode(1) == Some(" "));
        assert!(m.vocab["base"].decode(0).is_none());
        // cjk group is the largest script table (11k chars).
        assert!(m.vocab["cjk"].size() > 10_000);
        let cjk = m.vocab["cjk"].decode(2);
        assert_eq!(cjk, Some("!"));
        // lm_params per group (base has none).
        assert!(m.lm_params.contains_key("cjk"));
        assert!(m.lm_params.contains_key("world"));
        assert!(!m.lm_params.contains_key("base"));
    }
}
