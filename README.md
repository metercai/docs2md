# docs2md

Three-tier PDF → Markdown, one self-contained binary.

`docs2md` converts a PDF to Markdown by routing each page to the cheapest
tier that can read it:

| Tier | What it is | Where it runs | Cost |
|---|---|---|---|
| **L0** | native text extraction (embedded text layer) | in-process, no models | free |
| **L1** | **embedded** pure-Rust Indra OCR (ONNX on rten) + PP-OCRv6 G3 fallback | in-process, models baked in | ~0.7 s/page single-core (dense pages) |
| **L2** | external VLM service (a **llama-server wrapper**) | separate process, config-located | ~seconds/page |

L1's embedded models (the Indra ONNX pipeline and the PP-OCRv6-small
fallback) are `include_bytes!`-embedded, so the release is a single binary
with no external runtime (no Python, no dylib). The layout model (G1
routing) and the L2 VLM are deliberately external and located by config.

## Tiers, in detail

- **L0 — native extraction** (`pdf-inspector`). Pages with a usable text
  layer convert directly. No rendering, no models.
- **L1 — embedded dual engine** (pure Rust, ONNX on rten). The primary is
  the **Indra** OCR pipeline (`feature indra`, default on): detection,
  recognition, reading order and markdown assembly with a table
  post-filter. When Indra's output fails the **G3** output-level gate
  (short output and low confidence), the page falls back to the embedded
  **PP-OCRv6-small** once; if that output fails the gate again the page
  escalates to L2. `[l1] use_indra = false` runs the PP-OCRv6-only flow
  (the v0.1.0 behavior) with no fallback.
- **L2 — external VLM service** (`llama-server` wrapper). Used for complex
  pages (tables, figures, equations). `docs2md` talks to it over the
  OpenAI-compatible `/v1/chat/completions` API, sending the rendered page as
  a base64 image. The service is launched/supervised by the `docs2md-l2d`
  sub-project.

## Routing (per page)

```
page has a usable text layer ──────────────── L0 (direct)
needs OCR → render (pdf_oxide, 200 dpi)
          → G1 layout model (external PP-DocLayout):
              table-caption / compact complex / dense-text blob ⇒ L2
              otherwise                                        ⇒ L1
          → if L1 (use_indra, default):
              Indra → G3 output-level gate (markdown length, mean conf)
                  pass ⇒ L1-indra
                  fail ⇒ PP-OCRv6 (once) → same gate
                      pass ⇒ L1-ppv6(g3-fallback)
                      fail ⇒ escalate to L2 (L1->L2(g3+g2:{tag}))
          → if L1 (use_indra = false): the v0.1.0 PP-OCRv6-only flow
```

- **G1** (layout) is skipped when no `[l1] layout_model` is configured or it
  fails to load — then every OCR page defaults to L1 and G3 can still
  escalate to L2.
- **L2 unavailable** (no `base_url`, health probe failed, or the call
  failed): the L1 output is kept (or an empty page) and the page is marked
  `<!-- needs a higher-tier OCR model: L2 service unavailable -->`.

## Quick start

```sh
# 1. build (one binary; L1 small model is embedded)
cargo build --release
#    → target/release/docs2md

# 2. (optional) fetch the external layout model for G1 routing
./scripts/fetch-models.sh          # → models/DocOCR/pp_doclayout_plus_l.onnx

# 3. (optional) start the L2 service — a llama-server wrapper.
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

With no config file at all: L0 pages convert natively, OCR pages run the
embedded L1, and (since L2 is unconfigured) any page that needs a higher tier
is marked as such.

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
heading/table where the G1 layout model marks them); L1 (Indra) pages get
reading-ordered blocks from the order model's class head; L2 pages carry an
empty list.

## Configuration

See [`config.example.toml`](config.example.toml). All fields are defaulted;
unknown keys are a hard error. The config is the single place that locates
the external pieces (layout model path, L2 `base_url`). Routing thresholds
(`[route]`, G3 `min_chars`/`hard_conf`/`mid_conf`, G1 area/conf gates) are
configurable operational parameters.

## Repository layout

```
src/                 the docs2md binary + lib (orchestrator, L0/L1/L2, routing)
  config.rs          config model (serde, all defaulted, deny_unknown_fields)
  l0.rs              L0 native extraction (pdf-inspector)
  l1.rs              L1 engine trait + embedded Indra + PP-OCRv6 fallback
  l2.rs              L2 client (OpenAI-compatible /v1/chat/completions)
  render.rs          pdf_oxide rendering + pixel→PDF coordinate mapping
  route.rs           G1 layout routing + G3 output-level gate (+ tests)
  markdown.rs        OCR-line → markdown assembly (reading order)
  lib.rs             Engine orchestrator (per-page tier dispatch)
crates/ocr/          vendored rusto-rs core (PP-OCRv6, pure Rust, rten)
services/docs2md-l2d/ the L2 service wrapper (llama-server supervisor)
assets/models/small/ embedded PP-OCRv6-small det/rec/dict (in the binary)
assets/models/indra/ embedded Indra pipeline: det/rec/order/orient/script
                     ONNX + LM tables (in the binary)
scripts/             fetch-models.sh (external layout model + optional tiers)
```

## L1 "dual mode"

`[l1] tier` selects the PP-OCRv6 model tier (used by the G3 fallback and
by the `use_indra = false` flow). **`small` is embedded** (the default,
"one binary goes everywhere"). `medium`/`tiny` are *not* embedded;
selecting one requires `[l1] model_dir` pointing at an external
`PPOCR_v6_<tier>_rten/` directory (see `scripts/fetch-models.sh`).

## The Indra engine

The Indra pipeline (detection, recognition, reading order, markdown with
table post-filter) is embedded under `assets/models/indra/` behind the
`indra` feature (default on) and is the primary L1 engine: `[l1]
use_indra` is `true` by default. The PP-OCRv6 engine remains as the G3
fallback, so a page whose Indra output is short and low-confidence is
re-read once by PP-OCRv6 before it escalates to L2. `use_indra = false`
(or building with `--no-default-features`) returns to the PP-OCRv6-only
flow.

## L2 service model

The L2 tier is a **llama-server wrapper**, not a bespoke service:
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
healthy L2 service (a previous l2d, or a manually started llama-server)
it is reused and no second VLM process is started; a still-loading one
(503) is waited on within `health_timeout_secs` and then reused; an
occupant that never becomes healthy is a hard error with the diagnosis.

## Performance and size (measured, P9)

On the 623-page fixture corpus (8 PDFs), release build, this machine
(64-core):

| Metric | Reading | Result |
|---|---|---|
| L1 (pure `ocr_page`, 520 dense patent pages) | median, single core (`RTEN_NUM_THREADS=1`, CPU-pinned) | **0.66 s/page** (p90 1.02, max 1.55) |
| L1 vs the Python Indra reference | same dense pages, single core vs 4-thread wall (median 0.376 s) | 1.8x — inside the 2x rten budget |
| Full pipeline incl. PDF parse + render + G1 | 520 dense pages, 4 threads | ~2.0 s/page |
| Terminal-state histogram | 623 pages, layout model on, L2 unconfigured | S0=100 / S1a=216 / S1b=3 / S2=304 (plan: 100/216/3/304, k=0) |
| Determinism | two consecutive 623-page runs | byte-identical markdown |
| Binary | x86_64 release | ~127 MB raw (Indra pipeline 72 MB + PP-OCRv6-small + code); the 44 MB LM tables are the largest single block |

The binary is `include_bytes!`-embedded: no Python, no dylib, no fetched
model needed for L0/L1. Thread count is the process env var
`RTEN_NUM_THREADS` (default: physical core count) — lower it to trade
throughput for determinism or to fit a cgroup quota.

## License

MPL-2.0 (the project). The vendored OCR core (`crates/ocr/`) is MIT
(`rusto-rs`, byrizki). Model weights are distributed under their respective
licenses and are fetched separately (not in the repo).
