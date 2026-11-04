# docs2md

Three-tier PDF → Markdown, one self-contained binary.

`docs2md` converts a PDF to Markdown by routing each page to the cheapest
tier that can read it:

| Tier | What it is | Where it runs | Cost |
|---|---|---|---|
| **L0** | native text extraction (embedded text layer) | in-process, no models | free |
| **L1** | **embedded** pure-Rust PP-OCRv6 (ONNX on rten) | in-process, model baked in | ~1 page of CPU |
| **L2** | external VLM service (a **llama-server wrapper**) | separate process, config-located | ~seconds/page |

L1's PP-OCRv6-**small** model (≈31 MB) is `include_bytes!`-embedded, so the
release is a single binary with no external runtime (no Python, no dylib).
The layout model (G1 routing) and the L2 VLM are deliberately external and
located by config.

## Tiers, in detail

- **L0 — native extraction** (`pdf-inspector`). Pages with a usable text
  layer convert directly. No rendering, no models.
- **L1 — embedded PP-OCRv6-small** (`rusto-rs`, vendored, pure Rust). Used
  for simple scanned-text pages. The engine is behind an `L1Engine` trait;
  an **Indra** engine is a reserved slot (`models/indra/`, feature `indra`)
  that will implement the same trait once its ONNX pipeline is ported to
  Rust.
- **L2 — external VLM service** (`llama-server` wrapper). Used for complex
  pages (tables, figures, equations). `docs2md` talks to it over the
  OpenAI-compatible `/v1/chat/completions` API, sending the rendered page as
  a base64 image. The service is launched/supervised by the `docs2md-l2d`
  sub-project.

## Routing (per page)

```
page has a usable text layer ──────────────── L0 (direct)
needs OCR → render (pdf_oxide, 150 dpi)
          → G1 layout model (external PP-DocLayout):
              table-caption / compact complex / dense-text blob ⇒ L2
              otherwise                                        ⇒ L1
          → if L1: G2 output-level gate (chars, mean line conf)
              hard/medium ⇒ escalate to L2
```

- **G1** (layout) is skipped when no `[l1] layout_model` is configured or it
  fails to load — then every OCR page defaults to L1 and G2 can still
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

# 3. (optional) start the L2 service (a llama-server wrapper)
cargo run -p docs2md-l2d -- \
  --model /path/PaddleOCR-VL-1.6.gguf \
  --mmproj /path/PaddleOCR-VL-1.6-mmproj.gguf \
  --port 8080
#    prints: L2_READY http://127.0.0.1:8080

# 4. configure (optional; defaults run without a file)
cp config.example.toml docs2md.toml
#    set [l1] layout_model and [l2] base_url as needed

# 5. convert
target/release/docs2md input.pdf out.md
#    --pages 1,5,9   process only those pages
#    --no-markers    omit <!-- page:N --> markers
```

With no config file at all: L0 pages convert natively, OCR pages run the
embedded L1, and (since L2 is unconfigured) any page that needs a higher tier
is marked as such.

## Configuration

See [`config.example.toml`](config.example.toml). All fields are defaulted;
unknown keys are a hard error. The config is the single place that locates
the external pieces (layout model path, L2 `base_url`). Routing thresholds
(`[route]`, G2 `min_chars`/`hard_conf`/`mid_conf`, G1 area/conf gates) are
configurable operational parameters.

## Repository layout

```
src/                 the docs2md binary + lib (orchestrator, L0/L1/L2, routing)
  config.rs          config model (serde, all defaulted, deny_unknown_fields)
  l0.rs              L0 native extraction (pdf-inspector)
  l1.rs              L1 engine trait + embedded PP-OCRv6-small + Indra slot
  l2.rs              L2 client (OpenAI-compatible /v1/chat/completions)
  render.rs          pdf_oxide rendering + pixel→PDF coordinate mapping
  route.rs           G1 layout routing + G2 output-level gate (+ tests)
  markdown.rs        OCR-line → markdown assembly (reading order)
  lib.rs             Engine orchestrator (per-page tier dispatch)
crates/ocr/          vendored rusto-rs core (PP-OCRv6, pure Rust, rten)
services/docs2md-l2d/ the L2 service wrapper (llama-server supervisor)
assets/models/small/ embedded PP-OCRv6-small det/rec/dict (in the binary)
models/indra/        reserved slot for a future embedded Indra engine
scripts/             fetch-models.sh (external layout model + optional tiers)
```

## L1 "dual mode"

`[l1] tier` selects the PP-OCRv6 model tier. **`small` is embedded** (the
default, "one binary goes everywhere"). `medium`/`tiny` are *not* embedded;
selecting one requires `[l1] model_dir` pointing at an external
`PPOCR_v6_<tier>_rten/` directory (see `scripts/fetch-models.sh`). The engine
code is identical across tiers — only the model weights differ.

## The reserved Indra slot

`models/indra/` and the `indra` feature mark where the Indra engine (a
faster ONNX OCR pipeline) will be embedded once its Python pipeline is
ported to Rust. It is intentionally *not* implemented in v1: selecting
`[l1] use_indra = true` returns a clear "not ported yet" error rather than
silently degrading.

## L2 service model

The L2 tier is a **llama-server wrapper**, not a bespoke service:
`services/docs2md-l2d` supervises a `llama-server` process (a PaddleOCR-VL
GGUF), waits for its `/health` endpoint, and keeps it running. `docs2md`
consumes the OpenAI-compatible API. This means any OpenAI-compatible VLM
endpoint (self-hosted or remote) can be pointed at via `[l2] base_url`.

## License

MPL-2.0 (the project). The vendored OCR core (`crates/ocr/`) is MIT
(`rusto-rs`, byrizki). Model weights are distributed under their respective
licenses and are fetched separately (not in the repo).
