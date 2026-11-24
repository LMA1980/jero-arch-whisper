---
name: jero-arch-whisper
description: Encoder-decoder automatic speech recognition recipe (`WhisperRecipe`) with 16kHz mel-spectrogram Conv1d stem and cross-attention KV cache for the SpacemiT K3 platform.
---

# `jero-arch-whisper` Architecture Crate

Provides Encoder-decoder Whisper ASR recipe (`WhisperRecipe`) with 16kHz PCM/log-mel Conv1d stem, cached encoder cross-attention KV, and autoregressive decoder for K3.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
