//! docs2md CLI.
//!
//! ```text
//! docs2md [options] <input.pdf> [output.md]
//!
//! options:
//!   -c, --config <path>   config file (default: $DOCS2MD_CONFIG or ./docs2md.toml)
//!   -o, --out <path>      output markdown (default: <input>.md)
//!       --pages 1,5,9     process only these 1-based pages
//!       --no-markers      omit the per-page <!-- page:N --> markers
//!       --blocks <path>   also write structured layout blocks as JSON
//!   -V, --version         print version
//! ```

use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Args {
    config: Option<PathBuf>,
    out: Option<String>,
    pages: Option<Vec<u32>>,
    no_markers: bool,
    blocks: Option<PathBuf>,
    input: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        config: None,
        out: None,
        pages: None,
        no_markers: false,
        blocks: None,
        input: PathBuf::new(),
    };
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "-V" | "--version" => {
                println!("docs2md {VERSION}");
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-c" | "--config" => {
                a.config = Some(PathBuf::from(val(&raw, &mut i, "-c/--config")?));
            }
            "-o" | "--out" => {
                a.out = Some(val(&raw, &mut i, "-o/--out")?);
            }
            "--pages" => {
                let v = val(&raw, &mut i, "--pages")?;
                a.pages = Some(parse_pages(&v)?);
            }
            "--no-markers" => {
                a.no_markers = true;
            }
            "--blocks" => {
                let v = val(&raw, &mut i, "--blocks")?;
                a.blocks = Some(PathBuf::from(v));
            }
            other => {
                if a.input.as_os_str().is_empty() {
                    a.input = PathBuf::from(other);
                } else {
                    a.out = Some(other.to_string());
                }
            }
        }
        i += 1;
    }
    if a.input.as_os_str().is_empty() {
        return Err("missing input PDF (usage: docs2md [options] <input.pdf> [output.md])".into());
    }
    Ok(a)
}

fn val(raw: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    raw.get(*i)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_pages(s: &str) -> Result<Vec<u32>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|e| format!("bad page '{p}': {e}"))
        })
        .collect()
}

fn print_help() {
    println!("docs2md {VERSION} — three-tier PDF → Markdown (L0 native / L1 PP-OCRv6 / L2 VLM service)\n");
    println!("Usage: docs2md [options] <input.pdf> [output.md]\n");
    println!("Options:");
    println!("  -c, --config <path>   config file (default: $DOCS2MD_CONFIG or ./docs2md.toml)");
    println!("  -o, --out <path>      output markdown (default: <input>.md)");
    println!("      --pages 1,5,9     process only these 1-based pages");
    println!("      --no-markers      omit per-page <!-- page:N --> markers");
    println!("      --blocks <path>   also write structured layout blocks as JSON");
    println!("  -V, --version         print version");
    println!("  -h, --help            print this help");
}

fn default_out(input: &Path) -> String {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "out".into());
    format!("{stem}.md")
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let config_path = docs2md::resolve_config_path(args.config.as_ref());
    let mut cfg = match docs2md::config::Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    if args.no_markers {
        cfg.output.page_markers = false;
    }
    if !config_path.as_os_str().is_empty() && config_path.exists() {
        eprintln!("[config] {}", config_path.display());
    }

    let bytes = match std::fs::read(&args.input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read {}: {e}", args.input.display());
            std::process::exit(1);
        }
    };

    let mut engine = match docs2md::Engine::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    // D: an explicit --blocks also requests the optional L0 layout relabel
    // (a no-op unless [l1].layout_model points at a model that loads).
    if args.blocks.is_some() {
        engine.set_blocks_requested(true);
    }

    let result = match engine.convert(&bytes, args.pages.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let out_path = args.out.unwrap_or_else(|| default_out(&args.input));
    if let Some(parent) = Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    if let Err(e) = std::fs::write(&out_path, &result.markdown) {
        eprintln!("error: cannot write {out_path}: {e}");
        std::process::exit(1);
    }

    // --blocks <path>: structured layout blocks (the E1/E2 output surface),
    // one JSON object per page. Shape: {"page": N, "tier": "...", "blocks":
    // [{"label": "text|heading|table|...", "bbox": [x0,y0,x1,y1], "text":
    // "..."}]} — bbox normalized to [0,1] y-down, the same coordinate space
    // the layout judge consumes. L2-vlm pages carry an empty block list.
    if let Some(blocks_path) = &args.blocks {
        let pages_json: Vec<serde_json::Value> = result
            .pages
            .iter()
            .map(|p| {
                serde_json::json!({
                    "page": p.page,
                    "tier": p.tier,
                    "blocks": p.blocks.iter().map(|b| {
                        serde_json::json!({
                            "label": b.label,
                            "bbox": b.bbox,
                            "text": b.text,
                        })
                    }).collect::<Vec<_>>(),
                })
            })
            .collect();
        let doc = serde_json::json!({
            "input": args.input.display().to_string(),
            "pages": pages_json,
        });
        if let Some(parent) = blocks_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match std::fs::write(blocks_path, serde_json::to_string_pretty(&doc).unwrap()) {
            Ok(()) => eprintln!("[blocks] {} ({} pages)", blocks_path.display(), result.pages.len()),
            Err(e) => {
                eprintln!("error: cannot write {}: {e}", blocks_path.display());
                std::process::exit(1);
            }
        }
    }

    eprintln!(
        "[done] {} → {} ({} pages)",
        args.input.display(),
        out_path,
        result.pages.len()
    );
    if result.pages.iter().any(|p| p.note.is_some()) {
        let flagged: Vec<String> = result
            .pages
            .iter()
            .filter(|p| p.note.is_some())
            .map(|p| format!("p{} [{}]", p.page, p.tier))
            .collect();
        eprintln!(
            "[warn] {} page(s) need a higher-tier OCR model (L2 unavailable): {}",
            flagged.len(),
            flagged.join(", ")
        );
    }
}
