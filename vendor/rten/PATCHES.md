# Vendored rten 0.27.0 — patches

This directory is a vendored copy of [rten 0.27.0](https://crates.io/crates/rten)
(MIT OR Apache-2.0, see LICENSE-MIT / LICENSE-APACHE), patched for docs2md's
L1 inference performance. The vendored crate is wired in via
`[patch.crates-io] rten = { path = "vendor/rten" }` in the root Cargo.toml.

## Patch 1: conv_2d_pointwise — serial per-item GEMMs → one batched call

**File:** `src/ops/conv.rs`, function `conv_2d_pointwise`.

**Upstream behaviour (0.27.0):** the function runs a `for n in 0..batch` loop,
calling `gemm_uninit` once per batch item. Each per-item GEMM is small
(N = spatial dim, often a single 128-column block; M = output channels,
split into 2–4 row blocks), so under the default 64-thread global rayon pool
every one of the 64 items oversubscribes the pool. Measured on the rec
workload (batch=64, per-item [128,256]x[256,118]):

```
serial  (RAYON=1)   : 6.9 ms
serial  (RAYON=64)  : 23.3 ms   <- 3.4x SLOWER with more threads
batched (RAYON=64)  : 0.75 ms   <- 31x vs serial at 64 threads
```

**Patch:** build `a_list` (batch copies of the same packed kernel matrix) and
`b_list` (batch per-item input views), then call
`gemm.batched_gemm_uninit(out_data, &a_list, &b_list, opts)` once. The
rten-gemm `batched_gemm_uninit` runs `a.par_iter()` over the batch, so the
64 small GEMMs execute concurrently in a single rayon parallel region —
no per-item spawn/join round trips, no oversubscription.

**Numerical semantics: unchanged.** Same A, same per-item B, same column
bias, same f32 accumulation order within each item. The only change is the
scheduling of independent items. Product output (4-page A/B) is
byte-identical.

**S0 gate (2026-11-13):** product-level verification on 4 representative
pages (product-default threads): rec stage 5149ms → 3292ms (**-36.1%**,
gate was ≥10%); markdown byte-identical on all pages.

**Upstream status:** not yet fixed upstream. If rten lands an equivalent
change, re-vendor clean and delete this patch (and the
`[patch.crates-io]` section in the root Cargo.toml).

## How to re-vendor / upgrade rten

1. Copy the new crate source (e.g. from the cargo registry) over this
   directory, keeping the `src/`, `Cargo.toml`, `CHANGELOG.md`, `docs/`,
   `README.md` layout.
2. Restore LICENSE-MIT / LICENSE-APACHE if the crate package omits them.
3. Re-apply Patch 1 at the `conv_2d_pointwise` site (the serial
   `for n in 0..batch { gemm_uninit(...) }` loop is the stable anchor).
4. Re-run the S0 gate (4-page rec-stage A/B, ≥10% expected) and the 40-page
   byte-identity check before relying on it.
