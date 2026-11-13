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

## Patch 2: topk — serial per-lane sort → parallel lanes

**File:** `src/ops/reduce.rs`, function `topk`.

**Upstream behaviour (0.27.0):** a serial `for (values, (out_values,
indices)) in values.lanes(axis).zip(out_values.lanes_mut(axis))` loop, where
each lane runs a `select_nth_unstable` partial sort (O(axis_size)). In the rec
model the dominant TopK is k=8 over [7552, 1698] (axis_size=1698), so the loop
runs 7552 serial select_nth calls. Profiled at 89–132 ms per rec run
(30–31% of the rec stage) — the single largest rec op after the pointwise
GEMMs were fixed by Patch 1.

**Patch:** `values.lanes(axis).into_par_iter().zip(out_values.lanes(axis)
.zip(out_indices.lanes(axis)).into_par_iter()).for_each(...)`. Each lane is
independent (disjoint output lanes) and the per-lane select is unchanged, so
the result is bit-identical. `T` and the `Ops::topk` trait bound gain
`Send + Sync` (already implied by the existing `GemmInT`/`GemmOutT` bounds the
call sites use).

**Measured (b64 rec run):** TopK 89.1 ms → 11.8 ms (-86.7%).

**Upstream status:** not fixed upstream.

## Patch 3: depthwise_conv_2d — serial batch loop → nested rayon

**File:** `src/ops/conv/depthwise.rs`, function
`ConvExecutor::depthwise_conv_2d`.

**Upstream behaviour (0.27.0):** `for n in 0..batch { out_chans.into_par_iter()
... }` — `batch` *sequential* rayon regions, each parallel over its channels.
Per-item work is ~0.5–1.6 ms, so at batch=64 the serial per-item spawn/join
round trips add up. Profiled at 54.7 ms (k=9) + 13.2 ms (3x3) per b64 rec
run (15.6% of the stage).

**Patch:** the outer `for n in 0..batch` becomes `output.axis_iter_mut(0)
.into_par_iter().zip(input.axis_iter(0).into_par_iter()).for_each(|(mut
out_item, input_item)| { ... })`, keeping the inner `out_chans.into_par_iter()`
over channels. At batch=1 the task set is identical to upstream (all channels
parallel, no regression); at batch=64 the 8192 channel tasks are distributed
in a single wave instead of 64 sequential joins.

**Numerical semantics: unchanged.** Same per-(item, channel) work, same
accumulation. 48-page product A/B byte-identical.

**Measured (b64 rec run):** depthwise conv 54.7 ms → 13.7 ms (-75%).

**Upstream status:** not fixed upstream.

**Note (flat-flatten rejected):** an earlier variant flattened (batch,
channel) into one `0..batch*out_c` region. It was faster at high batch but
*regressed* small-batch pages (batch=1: 128 channel tasks serial inside one
task). The nested version is >= upstream at every batch size and was kept.

## How to re-vendor / upgrade rten

1. Copy the new crate source (e.g. from the cargo registry) over this
   directory, keeping the `src/`, `Cargo.toml`, `CHANGELOG.md`, `docs/`,
   `README.md` layout.
2. Restore LICENSE-MIT / LICENSE-APACHE if the crate package omits them.
3. Re-apply Patch 1 at the `conv_2d_pointwise` site (the serial
   `for n in 0..batch { gemm_uninit(...) }` loop is the stable anchor).
4. Re-run the S0 gate (4-page rec-stage A/B, ≥10% expected) and the 40-page
   byte-identity check before relying on it.
