---
name: jero-arch-whisper
description: Encoder-decoder automatic speech recognition recipe (`WhisperRecipe`) with 16kHz mel-spectrogram Conv1d stem and cross-attention KV cache for the SpacemiT K3 platform.
---

# `jero-arch-whisper` Architecture Crate

Provides the encoder-decoder Whisper ASR recipe (`WhisperRecipe`, `"whisper"`) with 16 kHz PCM/log-Mel preprocessing, 2-layer `Conv1d(k=3)` GELU stem, bidirectional encoder, cached per-layer encoder cross-attention KV (`[n_text_layer, 2, enc_len, n_text_state]`), and autoregressive causal text decoder.

## Public API & Capabilities

- **Presets & Metadata Hydration**: `WhisperVariant` (`Tiny`, `Base`, `Small`, `Medium`, `LargeV3Turbo`), `WhisperRecipe::{tiny, base, small, medium, large_v3_turbo, from_variant, from_metadata, from_gguf, from_safetensors}` supporting `n_mels = 80` (`v1`/`v2`) and `n_mels = 128` (`large-v3-turbo`).
- **Audio DSP Frontend**: `build_slaney_mel_filters`, `pcm_to_log_mel`, `pcm_to_log_mel_with_filters`, and `WhisperRecipe::pcm_to_log_mel` computing `[n_mels, n_frames]` log-Mel spectrograms from 16 kHz mono `f32` PCM using `filters.mel` or a deterministic Slaney Mel filterbank.
- **Audio Encoder & Cross-Attention KV Cache**: `WhisperRecipe::encode_audio` runs `encoder.conv1` (`stride=1`) -> `encoder.conv2` (`stride=2`) -> `encoder.positional_embedding` -> `0..n_audio_layer` bidirectional attention + GELU MLP blocks -> `encoder.ln_post`, and projects `(K_enc, V_enc)` once per decoder layer into `cross_attn_kv`.
- **Autoregressive Text Decoder (`ArchRecipe`)**: `prefill_step` (automatically running `encode_audio` when `ExecutionContext::audio_mel` and mutable `cross_attn_kv_mut` are attached) and `decode_step` (reusing cached `cross_attn_kv` alongside `KvCacheView` causal self-attention and tied/untied vocabulary projection).
- **ADR-045 Fail-Closed Invariant**: Strictly validates `ModelWeights` presence, shapes, supported quant formats, finiteness of inputs/weights, `cross_attn_kv` geometry, and token/context bounds without synthetic fallbacks.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (`ModelWeights`, `SafeTensorsFile`, `GgufFile`, `ExecutionContext`, `KvCacheView`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`, with portable scalar fallbacks when built with `--no-default-features`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
