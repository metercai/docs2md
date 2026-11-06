//! docs2md-l2d — the L2 service wrapper (a llama-server supervisor).
//!
//! The L2 tier is an *external* VLM service, and it is fundamentally a
//! **llama-server wrapper**: this binary supervises a `llama-server`
//! process (typically a PaddleOCR-VL GGUF), waits until it is healthy,
//! and keeps it running. The `docs2md` binary then talks to it over the
//! OpenAI-compatible `/v1/chat/completions` API (see `src/l2.rs`).
//!
//! # Configuration: file-based, all defaults, zero-config capable
//!
//! Like `docs2md`'s own `docs2md.toml`, l2d is configured by a TOML file
//! in which **every parameter has a default**. Config file search order
//! (first existing wins):
//!
//!   1. `--config <path>`        explicit (must exist)
//!   2. `$DOCS2MD_L2D_CONFIG`
//!   3. `./docs2md-l2d.toml`
//!   4. `~/.config/docs2md/l2d.toml`
//!   5. built-in defaults (no file needed at all)
//!
//! With no file anywhere, l2d runs on defaults: it will supervise
//! `llama-server` (from PATH) serving `models/PaddleOCR-VL-1.6.gguf` on
//! `127.0.0.1:8080`. The only thing a default deployment must still
//! provide is the ~1GB GGUF itself (see `scripts/fetch-models.sh`);
//! l2d fails fast with an actionable message if the model file is
//! missing.
//!
//! Invocation of llama-server is aligned to the official
//! PaddleOCR-VL-GGUF README (`-m <gguf> --mmproj <mm> --host --port
//! --temp 0`); no template flags — the official GGUF ships a llama.cpp
//! adapted jinja template inside the model, and that is the correct
//! default.
//!
//! Responsibilities of this supervisor:
//!   * fail fast if the model file is missing (the one asset a
//!     zero-config deployment must still fetch),
//!   * start idempotently: if the configured port already serves a
//!     healthy L2 service, reuse it and exit (no second VLM process);
//!     a non-healthy occupant is a hard error with a diagnosis,
//!   * spawn `llama-server` with the configured model + mmproj + port,
//!   * poll `{base}/health` until 200 (llama-server answers 503
//!     "Loading model" while the VLM loads — that is kept waiting),
//!   * report the ready URL to stdout (`L2_READY http://<host>:<port>`)
//!     so a launcher can wire it into docs2md's `[l2] base_url`,
//!   * forward SIGTERM/SIGINT to the child (no orphaned llama-server
//!     holding the port), and wait on the child.
//!
//! It does NOT do OCR itself — it only keeps the VLM process alive.

use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

use libc::c_int;
use serde::Deserialize;

// ── Defaults (llama-server conventions + this repo's model layout) ──────

/// llama-server's own defaults, used here as l2d's.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_CLI: &str = "llama-server";
/// Model layout produced by scripts/fetch-models.sh (./models). The official
/// GGUF release name, so a fresh machine + fetch + no config "just works".
const DEFAULT_MODEL: &str = "models/PaddleOCR-VL-1.6.gguf";
const DEFAULT_MMPROJ: &str = "models/PaddleOCR-VL-1.6-mmproj.gguf";
/// CPU load of a ~1GB VLM can take a while; fail-fast on a dead child is
/// handled separately (early-exit detection), so a generous budget is safe.
const DEFAULT_HEALTH_TIMEOUT_SECS: u64 = 300;

// ── Config ───────────────────────────────────────────────────────────────

/// All l2d parameters. Every field has a default (struct-level
/// `serde(default)` + per-field defaults), so a missing or empty file
/// runs on the full default set. Unknown keys are a hard error (typos
/// fail loudly), mirroring docs2md's main config.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L2dConfig {
    /// The VLM GGUF model. Relative paths resolve against the config
    /// file's directory (or CWD when using built-in defaults).
    #[serde(default = "L2dConfig::default_model")]
    pub model: String,
    /// The multimodal projector GGUF.
    #[serde(default = "L2dConfig::default_mmproj")]
    pub mmproj: String,
    /// llama-server bind address.
    #[serde(default = "L2dConfig::default_host")]
    pub host: String,
    /// llama-server bind port.
    #[serde(default = "L2dConfig::default_port")]
    pub port: u16,
    /// llama-server binary (PATH name or absolute path).
    #[serde(default = "L2dConfig::default_cli")]
    pub cli: String,
    /// CPU threads for llama-server. None = llama-server's own default.
    pub threads: Option<usize>,
    /// Override the chat template (llama-server `--chat-template-file`).
    /// None = the model's built-in template — the correct default for the
    /// official PaddleOCR-VL GGUF.
    pub chat_template_file: Option<String>,
    /// Override the mtmd image marker (env LLAMA_MEDIA_MARKER).
    /// None = llama-server's own handling.
    pub media_marker: Option<String>,
    /// Max seconds to wait for the /health probe (model load).
    #[serde(default = "L2dConfig::default_health_timeout")]
    pub health_timeout_secs: u64,
    /// Restart the child if it exits non-zero.
    pub restart: bool,
    /// Extra args appended verbatim to the llama-server invocation.
    pub extra: Vec<String>,
}

impl L2dConfig {
    fn default_model() -> String {
        DEFAULT_MODEL.to_string()
    }
    fn default_mmproj() -> String {
        DEFAULT_MMPROJ.to_string()
    }
    fn default_host() -> String {
        DEFAULT_HOST.to_string()
    }
    fn default_port() -> u16 {
        DEFAULT_PORT
    }
    fn default_cli() -> String {
        DEFAULT_CLI.to_string()
    }
    fn default_health_timeout() -> u64 {
        DEFAULT_HEALTH_TIMEOUT_SECS
    }

    /// Resolve a model-ish relative path against `base` (the config
    /// file's directory, or CWD for built-in defaults).
    fn resolve(&self, p: &str, base: &std::path::Path) -> PathBuf {
        let pb = PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            base.join(pb)
        }
    }
}

impl Default for L2dConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            mmproj: DEFAULT_MMPROJ.to_string(),
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            cli: DEFAULT_CLI.to_string(),
            threads: None,
            chat_template_file: None,
            media_marker: None,
            health_timeout_secs: DEFAULT_HEALTH_TIMEOUT_SECS,
            restart: false,
            extra: Vec::new(),
        }
    }
}

/// Candidate config file locations, in precedence order.
fn config_candidates() -> Vec<Option<PathBuf>> {
    // 1. --config is handled in main (it must exist; a miss is an error).
    vec![
        // 2. $DOCS2MD_L2D_CONFIG
        std::env::var("DOCS2MD_L2D_CONFIG").ok().map(PathBuf::from),
        // 3. ./docs2md-l2d.toml
        Some(PathBuf::from("docs2md-l2d.toml")),
        // 4. ~/.config/docs2md/l2d.toml
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/docs2md/l2d.toml")),
    ]
}

/// Load the config: first existing candidate wins; no candidate ⇒ pure
/// defaults (zero-config mode).
fn load_config(explicit: &Option<PathBuf>) -> Result<(L2dConfig, PathBuf), String> {
    if let Some(p) = explicit {
        let s = std::fs::read_to_string(p)
            .map_err(|e| format!("cannot read config {}: {e}", p.display()))?;
        let cfg: L2dConfig =
            toml::from_str(&s).map_err(|e| format!("invalid TOML in {}: {e}", p.display()))?;
        return Ok((cfg, p.clone()));
    }
    for p in config_candidates().into_iter().flatten() {
        if p.is_file() {
            let s = std::fs::read_to_string(&p)
                .map_err(|e| format!("cannot read config {}: {e}", p.display()))?;
            let cfg: L2dConfig =
                toml::from_str(&s).map_err(|e| format!("invalid TOML in {}: {e}", p.display()))?;
            eprintln!("[l2d] config: {}", p.display());
            return Ok((cfg, p));
        }
    }
    eprintln!("[l2d] no config file found — running on built-in defaults");
    Ok((L2dConfig::default(), PathBuf::from(".")))
}

// ── llama-server command (aligned to the official GGUF README) ──────────

fn build_command(cfg: &L2dConfig, base_dir: &std::path::Path) -> Command {
    let model = cfg.resolve(&cfg.model, base_dir);
    let mmproj = cfg.resolve(&cfg.mmproj, base_dir);
    // llama-server -m <gguf> --mmproj <mm> --host <ip> --port <n> --temp 0
    // No template flags: the official GGUF ships its own adapted template.
    let mut cmd = Command::new(&cfg.cli);
    cmd.arg("-m").arg(&model);
    cmd.arg("--mmproj").arg(&mmproj);
    cmd.arg("--host").arg(&cfg.host);
    cmd.arg("--port").arg(cfg.port.to_string());
    cmd.arg("--temp").arg("0");
    cmd.arg("--alias").arg("docs2md");
    if let Some(t) = cfg.threads {
        cmd.arg("--threads").arg(t.to_string());
    }
    if let Some(t) = &cfg.chat_template_file {
        let tp = cfg.resolve(t, base_dir);
        cmd.arg("--chat-template-file").arg(tp);
    }
    if let Some(marker) = &cfg.media_marker {
        cmd.env("LLAMA_MEDIA_MARKER", marker);
    }
    cmd.args(&cfg.extra);
    cmd
}

// ── Health probe (std-only raw HTTP GET) ─────────────────────────────────

/// Poll `{url}` until it answers 2xx, or until `timeout`. Non-2xx keeps
/// waiting — llama-server answers 503 "Loading model" while the VLM
/// loads, so that is the normal case during startup. If a `child` is
/// supplied and it has already exited, bail out early: its exit status
/// is the real diagnosis (bind failure, bad model path).
fn probe_healthy_until(
    url: &str,
    timeout: Duration,
    mut child: Option<&mut Child>,
) -> Result<(), String> {
    let start = Instant::now();
    let mut last = String::from("(no attempt yet)");
    loop {
        if let Some(c) = child.as_deref_mut() {
            if let Ok(Some(status)) = c.try_wait() {
                return Err(format!(
                    "llama-server exited during startup: {status} (last health: {last})"
                ));
            }
        }
        match health_get(url) {
            Ok(()) => return Ok(()),
            Err(e) => last = e, // 503 "Loading model" keeps waiting until timeout
        }
        if start.elapsed() > timeout {
            return Err(format!("service not healthy within {timeout:?}: {last}"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn wait_healthy(base: &str, timeout: Duration, child: &mut Child) -> Result<(), String> {
    probe_healthy_until(&format!("{base}/health"), timeout, Some(child))
}

/// Minimal HTTP GET that returns Ok on any 2xx. std::net on purpose: one
/// health GET does not need a dependency.
fn health_get(url: &str) -> Result<(), String> {
    let (host, port, path) = parse_url(url).ok_or_else(|| format!("bad health url: {url}"))?;
    let mut stream = std::net::TcpStream::connect((host, port))
        .map_err(|e| format!("connect {host}:{port}: {e}"))?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: docs2md-l2d/0.1\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    let head = String::from_utf8_lossy(&buf[..n]);
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

// ── Signal handling: no orphaned llama-server on the port ────────────────

// The child's pid, readable from the signal handler. The handler only does
// async-signal-safe things (atomics + kill(2) + signal/raise), no locks.
static CHILD_PID: AtomicI32 = AtomicI32::new(0);
static TERM: AtomicBool = AtomicBool::new(false);

extern "C" fn term_handler(sig: c_int) {
    unsafe {
        TERM.store(true, Ordering::SeqCst);
        let pid = CHILD_PID.load(Ordering::SeqCst);
        if pid > 0 {
            libc::kill(pid, libc::SIGTERM);
        }
        // Restore the default disposition and die by the signal itself:
        // the supervisor (or shell) sees l2d killed-by-signal, and the
        // child is already terminated.
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

/// On SIGTERM/SIGINT: forward the signal to llama-server, then re-raise for
/// ourselves. This prevents the orphan-llama-server-on-the-port failure
/// mode: without it, a kill/timeout of l2d leaves llama-server running and
/// holding the port, and the next start dies at bind.
fn install_signal_handler() {
    // SAFETY: term_handler is async-signal-safe (atomics, kill, signal,
    // raise); installed once before any other thread activity.
    unsafe {
        // libc::signal takes sighandler_t (= usize on this target).
        let handler: extern "C" fn(c_int) = term_handler;
        libc::signal(libc::SIGTERM, handler as usize);
        libc::signal(libc::SIGINT, handler as usize);
    }
}

/// True if something already listens on (host, port).
fn port_in_use(host: &str, port: u16) -> bool {
    std::net::TcpStream::connect((host, port)).is_ok()
}

// ── CLI (thin: config file is the source of truth) ───────────────────────

fn usage() -> &'static str {
    r#"docs2md-l2d — supervises a llama-server (L2 VLM) for docs2md.

Configuration is a TOML file; every parameter has a default, so with no
file at all it runs on built-in defaults (models/PaddleOCR-VL-1.6.gguf on
127.0.0.1:8080 via `llama-server` from PATH). Search order:

  1. --config <path>              explicit (must exist)
  2. $DOCS2MD_L2D_CONFIG
  3. ./docs2md-l2d.toml
  4. ~/.config/docs2md/l2d.toml
  5. built-in defaults

See services/docs2md-l2d/l2d.example.toml for every key.

Usage: docs2md-l2d [--config <path>]

Starting is idempotent: if the configured port already serves a healthy
L2 service (a previous l2d, or a manually started llama-server), it is
reused and no new process is started.

The ready URL is printed to stdout as: L2_READY http://<host>:<port>
"#
}

fn parse_cli() -> Result<Option<PathBuf>, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut config: Option<PathBuf> = None;
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
                println!("{}", usage());
                std::process::exit(0);
            }
            "--config" => config = Some(PathBuf::from(next(&mut i)?)),
            other => return Err(format!("unknown arg '{other}' (see --help)")),
        }
        i += 1;
    }
    Ok(config)
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    let cli_config = match parse_cli() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    let (cfg, cfg_file) = match load_config(&cli_config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    // Resolve model paths and fail fast if the model is missing: with
    // zero config this is the one thing a fresh machine must fetch.
    let model = cfg.resolve(&cfg.model, &cfg_file);
    let mmproj = cfg.resolve(&cfg.mmproj, &cfg_file);
    if !model.is_file() {
        eprintln!("[l2d] model not found: {}", model.display());
        eprintln!("[l2d]   fetch it (scripts/fetch-models.sh) or set `model` in the config file");
        std::process::exit(1);
    }
    if !mmproj.is_file() {
        eprintln!("[l2d] mmproj not found: {}", mmproj.display());
        eprintln!("[l2d]   fetch it alongside the model, or set `mmproj` in the config file");
        std::process::exit(1);
    }

    let base = format!("http://{}:{}", cfg.host, cfg.port);

    install_signal_handler();

    // Idempotent start (plan C): if the configured port already serves a
    // healthy L2 service, reuse it and exit — do not start a second VLM.
    // A second instance would only double ~1GB of RAM while the first
    // holds the port; and docs2md talks to base_url, not to any process,
    // so a healthy service on the configured port is by definition what
    // it needs. A healthy occupant is reused immediately; a still-loading
    // one (503) is given its full health budget to finish; an occupant
    // that never becomes healthy is a hard error with the diagnosis.
    if port_in_use(&cfg.host, cfg.port) {
        match probe_healthy_until(
            &format!("{base}/health"),
            Duration::from_secs(cfg.health_timeout_secs),
            None,
        ) {
            Ok(()) => {
                eprintln!("[l2d] {base} already serves a healthy L2 service — reusing it (idempotent start, no new process)");
                println!("L2_READY {base}");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!(
                    "[l2d] port {} is already in use, and {base}/health never became 2xx: {e}",
                    cfg.port
                );
                eprintln!(
                    "[l2d]   the occupant may still be loading the model (503), or a stale instance holds the port;"
                );
                eprintln!(
                    "[l2d]   wait longer (raise `health_timeout_secs`), or stop it first (e.g. pkill -f llama-server), or change `port` in the config"
                );
                std::process::exit(1);
            }
        }
    }

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        eprintln!("[l2d] spawning {} (attempt {attempt})", cfg.cli);
        let mut child = match build_command(&cfg, &cfg_file).spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[l2d] failed to spawn {}: {e}", cfg.cli);
                eprintln!("[l2d]   is the llama-server binary on PATH? set `cli` in the config.");
                std::process::exit(1);
            }
        };
        CHILD_PID.store(child.id() as i32, Ordering::SeqCst);

        // Wait for health; bail out immediately if the child dies early
        // (bind failure, bad model path): its exit status is the diagnosis.
        if let Err(e) = wait_healthy(
            &base,
            Duration::from_secs(cfg.health_timeout_secs),
            &mut child,
        ) {
            eprintln!("[l2d] {e}");
            let _ = child.kill();
            let _ = child.wait();
            CHILD_PID.store(0, Ordering::SeqCst);
            if cfg.restart {
                eprintln!("[l2d] restarting...");
                continue;
            }
            std::process::exit(1);
        }
        eprintln!("[l2d] {base} is healthy; supervisor running");
        println!("L2_READY {base}");

        // Supervise: poll for child exit (200ms) or a TERM flag.
        let status = loop {
            if TERM.load(Ordering::SeqCst) {
                std::process::exit(143);
            }
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => std::thread::sleep(Duration::from_millis(200)),
                Err(e) => {
                    eprintln!("[l2d] wait error: {e}");
                    break ExitStatus::default();
                }
            }
        };
        CHILD_PID.store(0, Ordering::SeqCst);
        eprintln!("[l2d] llama-server exited: {status}");
        if cfg.restart && !status.success() {
            eprintln!("[l2d] restart=true: relaunching...");
            continue;
        }
        std::process::exit(if status.success() { 0 } else { 1 });
    }
}
