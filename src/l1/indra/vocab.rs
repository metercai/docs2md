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
    /// The vocab holds multi-character (LaTeX command) tokens.
    pub has_commands: bool,
    /// The vocab holds Hangul jamo (syllables are read as jamo).
    pub jamo: bool,
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

    /// Decode a collapsed id sequence to text (port of vocab.Vocab.decode):
    /// blank/invalid ids are skipped; a jamo vocab composes the whole line
    /// (NFC); a command vocab re-inserts the one space a LaTeX command needs
    /// before a following letter ("\\alpha" + "x" must not read back as
    /// "\\alphax").
    pub fn decode_seq(&self, ids: &[i32]) -> String {
        let toks: Vec<&str> = ids
            .iter()
            .filter(|&&i| (1..=self.chars.len() as i32).contains(&i))
            .map(|&i| self.chars[(i - 1) as usize].as_str())
            .collect();
        if self.jamo {
            return nfc(&toks.join(""));
        }
        if !self.has_commands {
            return toks.join("");
        }
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            let nxt = toks.get(k + 1).copied().unwrap_or("");
            let cmd = t.len() > 2 && t.starts_with('\\') && t[1..].chars().all(is_alpha);
            let letter_next = nxt.chars().next().map(is_alpha).unwrap_or(false);
            if cmd && letter_next {
                out.push(' ');
            }
        }
        out
    }
}

/// str.isalpha (Unicode L*) for the single characters our vocabularies hold
/// — the decode_seq space rule must fire before letters (ASCII, Latin, Greek,
/// Cyrillic, Indic) and not before digits, braces or operators.
fn is_alpha(c: char) -> bool {
    if c.is_ascii_alphabetic() {
        return true;
    }
    let o = c as u32;
    (o == 0xB5)
        || (0xC0..=0xD6).contains(&o)
        || (0xD8..=0xF6).contains(&o)
        || (0x100..=0x24F).contains(&o) // Latin Extended A/B
        || (0x1E00..=0x1EFF).contains(&o) // Latin Extended Additional
        || (0x370..=0x3C9).contains(&o) // Greek (and Coptic)
        || (0x400..=0x484).contains(&o)
        || o == 0x487
        || (0x48A..=0x52F).contains(&o) // Cyrillic (incl. extended)
        || (0x904..=0x939).contains(&o) // Devanagari
        || (0x958..=0x961).contains(&o)
        || (0x986..=0x98C).contains(&o) // Bengali
        || (0x98F..=0x990).contains(&o)
        || (0x993..=0x9A8).contains(&o)
        || (0x9AA..=0x9B0).contains(&o)
}

/// NFC normalization for the character set these vocabularies hold: the only
/// composition is Hangul (measured: no other composing pair exists in any of
/// the four vocab tables). Canonical jamo (L 0x1100-0x115F, V 0x1161-0x1175,
/// T 0x11A8-0x11C2) compose greedily; compatibility jamo (0x3131-0x318E) do
/// not compose under Python's unicodedata either, so they pass through.
pub fn nfc(s: &str) -> String {
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28; // final consonants (index 0 = no final)
    const S0: u32 = 0xAC00;
    // Greedy composition state.
    #[derive(Clone, Copy)]
    enum H {
        L(u32),             // leading consonant, index
        LV(u32, u32),       // (leading, vowel)
        Lvt(u32, u32, u32), // (leading, vowel, final)
    }
    let syllable = |l: u32, v: u32, t: u32| -> char {
        char::from_u32(S0 + l * (V_COUNT * T_COUNT) + v * T_COUNT + t).unwrap_or('\u{FFFD}')
    };
    let mut out = String::new();
    let mut pending: Option<H> = None;
    let emit = |out: &mut String, p: Option<H>| match p {
        None => {}
        Some(H::L(l)) => out.push(char::from_u32(0x1100 + l).unwrap_or('\u{FFFD}')),
        Some(H::LV(l, v)) => out.push(syllable(l, v, 0)),
        Some(H::Lvt(l, v, t)) => out.push(syllable(l, v, t)),
    };
    for c in s.chars() {
        let o = c as u32;
        if (0x1100..=0x115F).contains(&o) {
            emit(&mut out, pending);
            pending = Some(H::L(o - 0x1100));
            continue;
        }
        let Some(p) = pending else {
            out.push(c);
            continue;
        };
        let vidx = (0x1161..=0x1175).contains(&o).then(|| o - 0x1161);
        let tidx = (0x11A8..=0x11C2).contains(&o).then(|| o - 0x11A7);
        match (p, vidx, tidx) {
            (H::L(l), Some(v), _) => {
                pending = Some(H::LV(l, v));
                continue;
            }
            (H::LV(l, v), _, Some(t)) => {
                pending = Some(H::Lvt(l, v, t));
                continue;
            }
            _ => {}
        }
        // Does not extend the syllable: flush, then re-feed this character.
        emit(&mut out, pending);
        if (0x1100..=0x115F).contains(&o) {
            pending = Some(H::L(o - 0x1100));
        } else {
            pending = None;
            out.push(c);
        }
    }
    emit(&mut out, pending);
    out
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
            let key = if g == "base" {
                "vocab"
            } else {
                &format!("vocab_{g}")
            };
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
            vocab.insert(
                g.to_string(),
                Vocab {
                    has_commands: chars.iter().any(|c| c.chars().count() > 1),
                    jamo: chars.iter().any(|c| c == "\u{1100}"),
                    chars,
                },
            );
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

    #[test]
    fn nfc_composes_hangul() {
        // LV composition (ground truth: Python unicodedata.normalize("NFC")).
        assert_eq!(nfc("\u{1100}\u{1161}"), "가");
        // LVT composition (11A8 = final index 1: index 0 is the empty slot).
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "각");
        // 0x11A7 is unassigned (not a T jamo): the syllable flushes and it
        // passes through.
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A7}"), "가\u{11A7}");
        // A second syllable starts after the first flushes.
        assert_eq!(nfc("\u{1100}\u{1161}\u{1102}\u{1165}"), "가너");
        assert_eq!(nfc("\u{1100}\u{1161}\u{1102}\u{1161}"), "가나");
        // Non-composing input passes through (a lone leading consonant,
        // a lone vowel, and the compatibility jamo do not compose).
        assert_eq!(nfc("\u{1100}"), "\u{1100}");
        assert_eq!(nfc("\u{1161}"), "\u{1161}");
        assert_eq!(nfc("\u{3131}\u{3164}"), "\u{3131}\u{3164}");
        assert_eq!(nfc("가\u{1100}나"), "가\u{1100}나");
    }

    #[test]
    fn decode_seq_latin_command_space() {
        let m = Meta::parse(META).unwrap();
        let v = &m.vocab["base"];
        let id = |tok: &str| v.chars.iter().position(|c| c == tok).unwrap() as i32 + 1;
        let a = id("a");
        let b = id("b");
        let frac = id("\\frac");
        let comma = id(",");
        // Plain characters concatenate.
        assert_eq!(v.decode_seq(&[a, b]), "ab");
        // A command before a letter re-inserts its space.
        assert_eq!(v.decode_seq(&[frac, a]), "\\frac a");
        // A command before a non-letter does not.
        assert_eq!(v.decode_seq(&[frac, comma]), "\\frac,");
        // Blanks and out-of-range ids are skipped.
        assert_eq!(v.decode_seq(&[0, a, 999999, b]), "ab");
    }

    #[test]
    fn decode_seq_cjk_composes_jamo() {
        let m = Meta::parse(META).unwrap();
        let v = &m.vocab["cjk"];
        let id = |tok: &str| v.chars.iter().position(|c| c == tok).unwrap() as i32 + 1;
        assert_eq!(
            v.decode_seq(&[
                id("\u{1100}"),
                id("\u{1161}"),
                id("\u{1102}"),
                id("\u{1161}")
            ]),
            "가나"
        );
    }
}
