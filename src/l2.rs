//! L2 client: talks to an external VLM service that is a **llama-server
//! wrapper** (e.g. `docs2md-l2d` supervising a `llama-server` running a
//! PaddleOCR-VL GGUF). We speak the OpenAI-compatible
//! `/v1/chat/completions` API, sending the page PNG as a base64 `image_url`
//! content part.
//!
//! L2 is *optional*: when `base_url` is empty or the health probe fails,
//! `available()` is false and the orchestrator falls back to L1, marking
//! the page "needs a higher-tier OCR model".

use crate::config::L2Config;

/// A page rendered to PNG (pixel, y-down) plus its 1-based page number.
pub struct PageImage<'a> {
    pub png: &'a [u8],
    pub page1: u32,
}

/// Outcome of one L2 call.
#[derive(Debug, Clone)]
pub struct L2Result {
    pub markdown: String,
    /// True when the model itself answered (vs. a transport failure).
    pub ok: bool,
    pub error: Option<String>,
}

/// L2 service client. Constructed once per run; `available` is resolved by
/// a single startup health probe (or by an empty `base_url`).
pub struct L2Client {
    cfg: L2Config,
    available: bool,
    /// Why the service is unavailable (for the provenance note). Empty when
    /// available.
    unavailable_reason: String,
}

impl L2Client {
    /// Build the client and (optionally) run the startup health probe.
    pub fn new(cfg: &L2Config) -> Self {
        let mut c = Self {
            cfg: cfg.clone(),
            available: false,
            unavailable_reason: String::new(),
        };
        if c.cfg.base_url.trim().is_empty() {
            c.unavailable_reason = "no L2 base_url configured".into();
            return c;
        }
        if c.cfg.health_check {
            match c.health() {
                Ok(()) => c.available = true,
                Err(e) => c.unavailable_reason = format!("health probe failed: {e}"),
            }
        } else {
            c.available = true;
        }
        c
    }

    /// GET `{base}/health`, retrying while llama-server reports it is still
    /// loading the model (503 `{"error":{"message":"Loading model"}}`).
    ///
    /// llama-server starts listening before the GGUF is resident in memory;
    /// during that window `/health` answers 503. A single probe right after
    /// the supervisor reports "ready" (which itself polls this same
    /// endpoint) can still race the tail of the load, so we retry within
    /// `health_wait_secs`: 5xx = keep waiting, 2xx = up, connection error =
    /// the service is gone, stop immediately.
    pub fn health(&self) -> Result<(), String> {
        let url = format!("{}/health", self.base());
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(self.cfg.health_wait_secs);
        loop {
            match self.probe_once(&url) {
                Probe::Up => return Ok(()),
                Probe::Loading => {}
                Probe::Down(reason) => return Err(reason),
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "L2 service still loading model after {}s",
                    self.cfg.health_wait_secs
                ));
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    /// One `/health` probe.
    fn probe_once(&self, url: &str) -> Probe {
        let res = ureq::get(url)
            .timeout(std::time::Duration::from_secs(5))
            .call();
        match res {
            Ok(r) => {
                // ureq 2.x returns non-2xx as Err(Status), so Ok means 2xx.
                let _ = r.into_string();
                Probe::Up
            }
            // llama-server answers 503 while the GGUF is still loading.
            Err(ureq::Error::Status(503, _)) => Probe::Loading,
            // Any other status/connection error: treat as down.
            Err(e) => Probe::Down(e.to_string()),
        }
    }

    pub fn available(&self) -> bool {
        self.available
    }

    pub fn unavailable_reason(&self) -> &str {
        &self.unavailable_reason
    }

    /// Send one rendered page to the VLM and return its markdown.
    pub fn ocr_page(&self, img: &PageImage<'_>) -> L2Result {
        if !self.available {
            return L2Result {
                markdown: String::new(),
                ok: false,
                error: Some(self.unavailable_reason.clone()),
            };
        }
        let url = format!("{}/v1/chat/completions", self.base());
        let b64 = base64_encode(img.png);
        let data_uri = format!("data:image/png;base64,{b64}");

        let model = self
            .cfg
            .model
            .clone()
            .unwrap_or_else(|| "default".into());

        let body = serde_json::json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": self.cfg.prompt },
                        { "type": "image_url", "image_url": { "url": data_uri } },
                    ],
                }
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.0,
        });

        let res = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.cfg.timeout_secs))
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| e.to_string());

        let res = match res {
            Ok(r) => r,
            Err(e) => {
                return L2Result {
                    markdown: String::new(),
                    ok: false,
                    error: Some(e),
                };
            }
        };

        let text = res.into_string().unwrap_or_default();
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                return L2Result {
                    markdown: String::new(),
                    ok: false,
                    error: Some(format!("non-JSON L2 response: {e} (body: {})",
                        text.chars().take(200).collect::<String>())),
                };
            }
        };

        // OpenAI-compatible shape: choices[0].message.content
        let content = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        if content.trim().is_empty() {
            // Surface an upstream error if present.
            let err = v
                .pointer("/error/message")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            return L2Result {
                markdown: String::new(),
                ok: false,
                error: err.or_else(|| Some("empty L2 content".to_string())),
            };
        }

        L2Result {
            markdown: clean_llama_output(&content),
            ok: true,
            error: None,
        }
    }

    fn base(&self) -> String {
        self.cfg
            .base_url
            .trim_end_matches('/')
            .to_string()
    }
}

/// Outcome of one `/health` probe.
enum Probe {
    /// 2xx: service up and model resident.
    Up,
    /// 5xx: service is up but not ready yet (loading model).
    Loading,
    /// Connection-level failure: service is gone.
    Down(String),
}

/// Strip a trailing llama-cli/llama-server status block if it leaks into the
/// content ("[ Prompt: ... ]", "> ", "Exiting..."). Harmless when absent.
fn clean_llama_output(s: &str) -> String {
    let mut s = s.to_string();
    if let Some(pos) = s.rfind("[ Prompt:") {
        if s[pos..].contains("Exiting...") {
            s.truncate(pos);
        }
    }
    while let Some(last) = s.strip_suffix(['\n', ' ', '>']) {
        s.truncate(last.len());
    }
    s
}

/// Minimal base64 encoder (no external dep; the payload is a PNG, a few MB
/// at most, so a dependency-free encoder keeps the binary lean).
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() / 3 + 1) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn empty_base_url_is_unavailable() {
        let c = L2Client::new(&L2Config {
            base_url: String::new(),
            ..Default::default()
        });
        assert!(!c.available());
        assert!(c.unavailable_reason().contains("no L2 base_url"));
    }

    #[test]
    fn clean_llama_output_strips_status_block() {
        assert_eq!(
            clean_llama_output("# Title\n\nbody\n"),
            "# Title\n\nbody"
        );
        // A leaked llama status block is truncated at its start.
        assert_eq!(
            clean_llama_output("body\n[ Prompt: OCR: ]\nExiting..."),
            "body"
        );
        // No trailing whitespace, no status block: unchanged.
        assert_eq!(clean_llama_output("abc"), "abc");
    }
}
