# docs2md

Tiered PDF → Markdown, one self-contained binary.

`docs2md` converts a PDF to Markdown by routing each page to the cheapest
tier that can read it:

| Tier | What it is | Where it runs | Cost |
|---|---|---|---|
| **L0** | native text extraction (embedded text layer) | in-process, no models | free |
| **L1** | **embedded** pure-Rust Indra OCR (ONNX on rten) | in-process, models baked in | ~0.7 s/page single-core (dense pages) |
| **L2** | **embedded** PP-OCRv6-small (reached only via the L1 serial chain) | in-process, models baked in | added cost on escalated pages |
| **L3** | external VLM service (a **llama-server wrapper**) | separate process, config-located | ~seconds/page |

The L1 and L2 engines (the Indra ONNX pipeline and the PP-OCRv6-small
fallback) are `include_bytes!`-embedded, so the release is a single binary
with no external runtime (no Python, no dylib). The layout model (optional
`--blocks` relabel) and the L3 VLM are deliberately external and located by
config.

## Tiers, in detail

- **L0 — native extraction** (`pdf-inspector`). Pages with a usable text
  layer convert directly (no rendering, no models). The coverage gate
  (`[route] coverage_gate`, default on) reroutes a page to L1 when its L0
  output covers less than `cov_theta` of the text layer — the extraction
  is broken (tables, multi-column pages the native layer flattens).
- **L1 — embedded OCR** (pure Rust, ONNX on rten). The primary is the
  **Indra** OCR pipeline (`feature indra`, default on): detection, a script
  pre-pass, recognition, reading order and markdown assembly with a table
  post-filter. A **serial metric chain** (`[route.chain]`) evaluates
  intermediate metrics against configured thresholds; the first stage whose
  criterion hits escalates the page to L2. `[l1] use_indra = false` runs
  the PP-OCRv6-only flow (the v0.1.0 behavior) with no Indra.
- **L2 — embedded PP-OCRv6-small** (pure Rust, ONNX on rten). Reached only
  through the serial chain — never by a pre-emptive layout detector. If its
  result is weak/invalid (`[l3].min_chars` / `[l3].hard`), the page
  escalates to the L3 VLM (when configured).
- **L3 — external VLM service** (`llama-server` wrapper). Used for complex
  pages (tables, figures, equations) that both embedded engines read
  weakly. `docs2md` talks to it over the
  OpenAI-compatible `/v1/chat/completions` API, sending the rendered page as
  a base64 image. The service is launched/supervised by the `docs2md-l2d`
  sub-project.

## Routing (per page)

```
page has a usable text layer ──────────────── L0 (direct), unless the
   coverage gate (default on) fires: L0 output covering less than
   [route] cov_theta of the text layer ⇒ reroute to L1
needs OCR → render (PDFium, 200 dpi)
          → L1 (embedded engine; `indra` default, PP-OCRv6-only when
            [l1].use_indra = false):
            det → script pre-pass → rec → order → markdown assembly
            with a serial metric chain (route.rs, [route.chain]):
                det lines < det_min_lines
                script cjk fraction above threshold
                (script unsure fraction: opt-in criterion, off by default)
                rec mean conf < rec_hard_conf
                assemble output chars < assemble.min_chars
                any stage hit ⇒ L2
          → L2 (embedded PP-OCRv6-small):
                weak/invalid result (output chars < [l3].min_chars or
                mean conf < [l3].hard) ⇒ L3 (VLM, when [l2].base_url set)
```

- **L3 unavailable** (no `base_url`, health probe failed, or the call
  failed): the best local output is kept (or the explicit low-confidence
  marker) and the page is marked accordingly — the cascade always ends with
  non-empty output (a page is never dropped).

- **Parallelism** (`[route] parallel_threads`, default 4): the page loop
  runs on worker threads, each with its own engine; output is reassembled
  in page order and is byte-identical to the serial run. Multi-page
  documents benefit (a 48-page doc: ~34.6 s at 4 workers vs ~72.9 s
  serial); single-page inputs are unaffected.

## Quick start

```sh
# 1. build (one binary; L1 small model is embedded)
cargo build --release
#    → target/release/docs2md

# 2. (optional) fetch the external layout model for the --blocks relabel
./scripts/fetch-models.sh          # → models/DocOCR/pp_doclayout_plus_l.onnx

# 3. (optional) start the L3 service — a llama-server wrapper.
#    l2d is config-file based with all-defaulted parameters:
#    if the GGUF sits under ./models (scripts/fetch-models.sh layout),
#    no config file is needed at all.
cargo run -p docs2md-l2d
#    or with a config file (see services/docs2md-l2d/l2d.example.toml):
#    docs2md-l2d --config ./docs2md-l2d.toml
#    prints: L2_READY http://127.0.0.1:8080

# 4. configure (optional; defaults run without a file)
cp config.example.toml docs2md.toml
#    set [l1] layout_model and [l2] base_url as needed

# 5. convert
target/release/docs2md input.pdf out.md
#    --pages 1,5,9   process only those pages
#    --no-markers    omit <!-- page:N --> markers
#    --blocks out.json  also write structured layout blocks as JSON
```

With no config file at all: L0 pages convert natively (the coverage gate
reroutes a page to L1 when its text layer is broken — tables, multi-column
layout), OCR pages run the embedded L1 (escalating weak pages to the
embedded L2), and — since the L3 VLM is unconfigured — a page whose L2
result is weak keeps the best local output with a note.

## Structured blocks (JSON)

`docs2md input.pdf out.md --blocks out.json` writes the per-page structured
layout blocks alongside the markdown — a new output surface that never
changes the markdown bytes or tier routing:

```json
{
  "input": "input.pdf",
  "pages": [
    {
      "page": 1,
      "tier": "L1-indra",
      "blocks": [
        { "label": "heading", "bbox": [0.1, 0.05, 0.9, 0.1], "text": "Title" },
        { "label": "table",   "bbox": [0.05, 0.2, 0.95, 0.6], "text": "…" }
      ]
    }
  ]
}
```

Each block carries a `label` from an 11-label vocabulary
(`text`, `heading`, `table`, `form`, `list`, …), a `bbox` normalized to
`[0,1]` y-down (the same coordinate space a layout evaluator consumes), and
the block `text`. L0 pages get text-layer line blocks (relabelled
heading/table where the layout model marks them, when `layout_model` is
configured); L1 (Indra) pages get reading-ordered blocks from the order
model's class head; L2 pages carry an empty list.

## Configuration

See [`config.example.toml`](config.example.toml). All fields are defaulted;
unknown keys are a hard error. The config is the single place that locates
the external pieces (layout model path, L3 `base_url`). The serial-chain
thresholds (`[route.chain]`), the L2→L3 criteria (`[l3]`), and the output
gate are configurable operational parameters.

## Repository layout

```
src/                 the docs2md binary + lib (orchestrator, L0/L1/L2/L3, routing)
  config.rs          config model (serde, all defaulted, deny_unknown_fields)
  l0.rs              L0 native extraction (pdf-inspector)
  l1.rs              L1 engine trait + embedded Indra + embedded PP-OCRv6
  l2.rs              L3 VLM client (OpenAI-compatible /v1/chat/completions)
  render.rs          PDFium (firecrawl) rendering + pixel→PDF coordinate mapping
  route.rs           the L1→L2 serial metric chain + L2→L3 criteria (+ tests)
  markdown.rs        OCR-line → markdown assembly (reading order)
  lib.rs             Engine orchestrator (per-page tier dispatch)
crates/ocr/          vendored rusto-rs core (PP-OCRv6, pure Rust, rten)
services/docs2md-l2d/ the L3 service wrapper (llama-server supervisor)
assets/models/small/ embedded PP-OCRv6-small det/rec/dict (in the binary)
assets/models/indra/ embedded Indra pipeline: det/rec/order/orient/script
                     ONNX + LM tables (in the binary)
scripts/             fetch-models.sh (external layout model + optional tiers)
```

## L2 engine tiers

`[l1] tier` selects the PP-OCRv6 model tier (used by the L2 fallback engine
and by the `use_indra = false` flow). **`small` is embedded** (the default,
"one binary goes everywhere"). `medium`/`tiny` are *not* embedded;
selecting one requires `[l1] model_dir` pointing at an external
`PPOCR_v6_<tier>_rten/` directory (see `scripts/fetch-models.sh`).

## The Indra engine

The Indra pipeline (detection, script pre-pass, recognition, reading order,
markdown with table post-filter) is embedded under `assets/models/indra/`
behind the `indra` feature (default on) and is the primary L1 engine:
`[l1] use_indra` is `true` by default. The embedded PP-OCRv6-small remains
as the L2 fallback engine, so a page whose Indra output trips the serial
chain is re-read once by PP-OCRv6 before it can escalate to the L3 VLM.
`use_indra = false` (or building with `--no-default-features`) returns to
the PP-OCRv6-only flow.

## L3 service model

The L3 tier is a **llama-server wrapper**, not a bespoke service:
`services/docs2md-l2d` supervises a `llama-server` process (a PaddleOCR-VL
GGUF), waits for its `/health` endpoint (503 "Loading model" is kept
waiting), and keeps it running — forwarding SIGTERM/SIGINT so no orphaned
llama-server is left holding the port. `docs2md` consumes the
OpenAI-compatible API. This means any OpenAI-compatible VLM endpoint
(self-hosted or remote) can be pointed at via `[l2] base_url`.

l2d itself is configured by a TOML file with **every parameter defaulted**
(search: `--config` > `$DOCS2MD_L2D_CONFIG` > `./docs2md-l2d.toml` >
`~/.config/docs2md/l2d.toml` > built-in defaults), mirroring docs2md's own
config: with the GGUF fetched into `./models`, a bare `docs2md-l2d`
starts with zero configuration.

Starting is **idempotent**: if the configured port already serves a
healthy L3 service (a previous l2d, or a manually started llama-server)
it is reused and no second VLM process is started; a still-loading one
(503) is waited on within `health_timeout_secs` and then reused; an
occupant that never becomes healthy is a hard error with the diagnosis.

## Performance and size (measured, P9)

On the 623-page fixture corpus (8 PDFs), release build, this machine
(64-core):

| Metric | Reading | Result |
|---|---|---|
| L1 (pure `ocr_page`, 520 dense patent pages) | median, single core (`RTEN_NUM_THREADS=1`) | **1.07 s/page** (p90 1.58, max 2.21) — up from the pre-script-pre-pass 0.66 s; the serial chain's script stage now classifies every line before recognition |
| L1 vs the Python Indra reference | same dense pages, single core vs 4-thread wall (median 0.376 s) | 2.8x — over the plan's 2x rten budget (t7: re-calibrate or trim the pre-pass) |
| Full pipeline incl. PDF parse + render | 623 pages, 4 threads | **2.25 s/page** (215 of them escalated to the embedded L2) |
| Terminal-state histogram | 623 pages, L3 unconfigured | **S0=85 / S1a=323 / S1b=215 / S2=0** (serial chain routes 215 pages to the embedded L2; none reach the VLM) |
| Determinism | two consecutive 623-page runs | byte-identical markdown |
| Binary | x86_64 release | 134.6 MB / 128.4 MiB (Indra pipeline 72 MB + PP-OCRv6-small + code); the 44 MB LM tables are the largest single block |

The binary is `include_bytes!`-embedded: no Python, no dylib, no fetched
model needed for L0/L1. Thread count is the process env var
`RTEN_NUM_THREADS` (default: physical core count) — lower it to trade
throughput for determinism or to fit a cgroup quota.

## License

MPL-2.0 (the project). The vendored OCR core (`crates/ocr/`) is MIT
(`rusto-rs`, byrizki). Model weights are distributed under their respective
licenses and are fetched separately (not in the repo).
