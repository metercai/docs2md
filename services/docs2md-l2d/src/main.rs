//! docs2md-l2d — the L2 service wrapper.
//!
//! The L2 tier is an *external* VLM service, and it is fundamentally a
//! **llama-server wrapper**: this binary supervises a `llama-server`
//! process (typically a PaddleOCR-VL GGUF), waits until it is healthy, and
//! keeps it running. The `docs2md` binary then talks to it over the
//! OpenAI-compatible `/v1/chat/completions` API (see `src/l2.rs`).
//!
//! Responsibilities of this supervisor:
//!   * spawn `llama-server` with the model + mmproj + port,
//!   * poll `{base}/health` until it answers (startup can take a while
//!     while the VLM loads),
//!   * report the ready URL to stdout (for a human or a launcher to wire
//!     into `docs2md`'s `[l2] base_url`),
//!   * forward signals so Ctrl-C stops the child cleanly.
//!
//! It does NOT do OCR itself — it only keeps the VLM process alive.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

struct Args {
    model: PathBuf,
    mmproj: PathBuf,
    host: String,
    port: u16,
    cli: PathBuf,
    threads: Option<usize>,
    /// Extra args appended verbatim to the llama-server invocation.
    extra: Vec<String>,
    health_timeout_secs: u64,
    /// Restart the child if it exits non-zero (default: false).
    restart: bool,
}

const DEFAULT_CLI: &str = "llama-server";

fn usage() -> String {
    "\
docs2md-l2d — supervises a llama-server (L2 VLM) for docs2md.

Usage: docs2md-l2d --model <gguf> --mmproj <mmproj.gguf> [options]

Options:
  --model <path>        the VLM GGUF model (e.g. PaddleOCR-VL-1.6.gguf)
  --mmproj <path>       the multimodal projector GGUF
  --host <ip>           bind address (default: 127.0.0.1)
  --port <n>            bind port (default: 8080)
  --cli <path>          llama-server binary (default: $L2_LLAMA_CLI or 'llama-server')
  --threads <n>         CPU threads for llama-server
  --health-timeout <s>  max seconds to wait for the /health probe (default: 120)
  --restart             restart the child if it exits (default: off)
  --extra <a b c>       extra llama-server args (everything after --extra)

The ready URL is printed to stdout as: L2_READY http://<host>:<port>
"
    .to_string()
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut a = Args {
        model: PathBuf::new(),
        mmproj: PathBuf::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        cli: PathBuf::new(),
        threads: None,
        extra: Vec::new(),
        health_timeout_secs: 120,
        restart: false,
    };
    let mut i = 1;
    while i < raw.len() {
        let flag = raw[i].as_str();
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--model" => a.model = PathBuf::from(next(&mut i)?),
            "--mmproj" => a.mmproj = PathBuf::from(next(&mut i)?),
            "--host" => a.host = next(&mut i)?,
            "--port" => a.port = next(&mut i)?.parse().map_err(|e| format!("bad --port: {e}"))?,
            "--cli" => a.cli = PathBuf::from(next(&mut i)?),
            "--threads" => {
                a.threads = Some(next(&mut i)?.parse().map_err(|e| format!("bad --threads: {e}"))?)
            }
            "--health-timeout" => {
                a.health_timeout_secs =
                    next(&mut i)?.parse().map_err(|e| format!("bad --health-timeout: {e}"))?
            }
            "--restart" => a.restart = true,
            "--extra" => {
                // consume all remaining args verbatim
                i += 1;
                while i < raw.len() {
                    a.extra.push(raw[i].clone());
                    i += 1;
                }
            }
            other => return Err(format!("unknown arg '{other}' (see --help)")),
        }
        i += 1;
    }
    if a.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    if a.mmproj.as_os_str().is_empty() {
        return Err("--mmproj is required".into());
    }
    if a.cli.as_os_str().is_empty() {
        a.cli = std::env::var("L2_LLAMA_CLI").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(DEFAULT_CLI)
        });
    }
    Ok(a)
}

fn build_command(a: &Args) -> Command {
    let mut cmd = Command::new(&a.cli);
    cmd.arg("-m").arg(&a.model);
    cmd.arg("--mmproj").arg(&a.mmproj);
    cmd.arg("--host").arg(&a.host);
    cmd.arg("--port").arg(a.port.to_string());
    cmd.arg("--alias").arg("docs2md");
    if let Some(t) = a.threads {
        cmd.arg("--threads").arg(t.to_string());
    }
    // Multimodal: accept images. llama-server auto-enables this when an
    // mmproj is loaded; we keep the default prompt off.
    cmd.args(&a.extra);
    cmd
}

fn wait_healthy(base: &str, timeout: Duration) -> Result<(), String> {
    let url = format!("{base}/health");
    let start = Instant::now();
    let mut last = String::from("(no attempt yet)");
    loop {
        match ureq_free_health(&url) {
            Ok(()) => return Ok(()),
            Err(e) => last = e,
        }
        if start.elapsed() > timeout {
            return Err(format!("service not healthy within {timeout:?}: {last}"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Minimal HTTP GET that returns Ok on any 2xx. Implemented with std::net to
/// keep the supervisor dependency-free (no ureq/reqwest needed for a single
/// health GET).
fn ureq_free_health(url: &str) -> Result<(), String> {
    let (host, port, path) = parse_url(url).ok_or_else(|| format!("bad health url: {url}"))?;
    let mut stream = std::net::TcpStream::connect((host, port))
        .map_err(|e| format!("connect {host}:{port}: {e}"))?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    let head = String::from_utf8_lossy(&buf[..n]);
    // "HTTP/1.1 200 OK" etc.
    let status = head
        .split_whitespace()
        .nth(1)
        .map(str::to_string)
        .unwrap_or_default();
    if status.starts_with('2') {
        Ok(())
    } else {
        Err(format!("status {status}"))
    }
}

fn parse_url(url: &str) -> Option<(&str, u16, &str)> {
    let rest = url.strip_prefix("http://")?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (h, p) = hostport.rsplit_once(':')?;
    Some((h, p.parse().ok()?, path))
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{}", usage());
            std::process::exit(2);
        }
    };
    let base = format!("http://{}:{}", a.host, a.port);

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        eprintln!("[l2d] spawning {} (attempt {attempt})", a.cli.display());
        let mut child = match build_command(&a).spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[l2d] failed to spawn {}: {e}", a.cli.display());
                std::process::exit(1);
            }
        };

        // Wait for health.
        match wait_healthy(&base, Duration::from_secs(a.health_timeout_secs)) {
            Ok(()) => {
                println!("L2_READY {base}");
                eprintln!("[l2d] {base} is healthy");
            }
            Err(e) => {
                eprintln!("[l2d] {e}");
                let _ = child.kill();
                let _ = child.wait();
                if a.restart {
                    eprintln!("[l2d] restarting...");
                    continue;
                }
                std::process::exit(1);
            }
        }

        // Wait for the child to exit.
        let status = child.wait().unwrap_or_default();
        eprintln!("[l2d] llama-server exited: {status}");
        if a.restart && status.success() == false {
            eprintln!("[l2d] --restart: relaunching...");
            continue;
        }
        std::process::exit(if status.success() { 0 } else { 1 });
    }
}
