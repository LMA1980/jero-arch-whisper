//! `jero-arch-whisper`: Encoder-decoder speech recognition recipe (`WhisperRecipe`) with 16 kHz
//! log-Mel `Conv1d` stem, cached encoder cross-attention KV, and autoregressive causal text decoder
//! for the `SpacemiT` K3 platform.

#![deny(unsafe_op_in_unsafe_fn)]

mod audio_dsp;
mod ops;
mod recipe;
mod weights;

pub use audio_dsp::{
    WHISPER_CHUNK_SECONDS, WHISPER_HOP_LENGTH, WHISPER_MAX_FRAMES, WHISPER_N_BINS_201,
    WHISPER_N_BINS_257, WHISPER_N_FFT, WHISPER_SAMPLE_RATE, build_slaney_mel_filters,
    pcm_to_log_mel, pcm_to_log_mel_with_filters,
};
pub use recipe::{
    WHISPER_EOT_TOKEN, WHISPER_NO_TIMESTAMPS_TOKEN_V2, WHISPER_NO_TIMESTAMPS_TOKEN_V3,
    WHISPER_SOT_TOKEN, WHISPER_TRANSCRIBE_TOKEN_V2, WHISPER_TRANSCRIBE_TOKEN_V3, WHISPER_VOCAB_V2,
    WHISPER_VOCAB_V3, WhisperRecipe, WhisperVariant,
};

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::gguf::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
    use jero_core::{
        ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
        SafeTensorsFile, TensorView,
    };

    struct OwnedTensor {
        name: String,
        quant: QuantType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    struct OwnedWeights {
        entries: Vec<OwnedTensor>,
    }

    impl OwnedWeights {
        fn remove(&mut self, name: &str) {
            self.entries.retain(|e| e.name != name);
        }

        fn replace(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
            if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
                entry.quant = quant;
                entry.shape = shape;
                entry.bytes = bytes;
            } else {
                self.entries.push(OwnedTensor {
                    name: name.to_owned(),
                    quant,
                    shape,
                    bytes,
                });
            }
        }
    }

    impl ModelWeights for OwnedWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            let entry = self
                .entries
                .iter()
                .find(|e| e.name == name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name}' not found"),
                })?;
            TensorView::from_bytes(&entry.name, entry.quant, &entry.shape, 0, &entry.bytes)
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.entries.iter().any(|e| e.name == name)
        }

        fn tensor_count(&self) -> usize {
            self.entries.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.entries.iter().map(|e| e.bytes.len()).sum()
        }
    }

    fn encode_f32(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0);
        let mant = bits & 0x007F_FFFF;
        if val == 0.0 {
            return sign;
        }
        let unbiased = exp - 127;
        let f16_exp = unbiased + 15;
        if f16_exp <= 0 {
            return sign;
        }
        if f16_exp >= 31 {
            return sign | 0x7C00;
        }
        let exp_u16 = u16::try_from(f16_exp).unwrap_or(0) << 10;
        let mant_u16 = u16::try_from(mant >> 13).unwrap_or(0);
        sign | exp_u16 | mant_u16
    }

    fn encode_f16(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        out
    }

    fn det_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let raw = (((i * 13 + seed * 7 + 5) % 23) as f32 - 11.0) * scale;
                raw
            })
            .collect()
    }

    fn det_ones_perturbed(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let delta = (((i + seed) % 5) as f32 - 2.0) * 0.05;
                1.0 + delta
            })
            .collect()
    }

    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        let len = u64::try_from(s.len()).expect("str len fits u64");
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_gguf_v3_with_payloads(
        arch: &str,
        kvs: &[(&str, u32)],
        tensors: &[(&str, QuantType, &[usize], &[u8])],
    ) -> Vec<u8> {
        let align = 32_usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747_u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&3_u32.to_le_bytes()); // v3
        buf.extend_from_slice(&(u64::try_from(tensors.len()).unwrap()).to_le_bytes());
        buf.extend_from_slice(&(u64::try_from(kvs.len() + 2).unwrap()).to_le_bytes());

        // general.architecture
        push_gguf_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8_u32.to_le_bytes()); // String
        push_gguf_str(&mut buf, arch);

        // general.alignment
        push_gguf_str(&mut buf, "general.alignment");
        buf.extend_from_slice(&4_u32.to_le_bytes()); // Uint32
        buf.extend_from_slice(&(u32::try_from(align).unwrap()).to_le_bytes());

        for &(k, v) in kvs {
            push_gguf_str(&mut buf, k);
            buf.extend_from_slice(&4_u32.to_le_bytes()); // Uint32
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let mut data_payload = Vec::new();
        for &(name, quant, shape, bytes) in tensors {
            let rem = data_payload.len() % align;
            if rem != 0 {
                data_payload.resize(data_payload.len() + (align - rem), 0);
            }
            let rel_off = u64::try_from(data_payload.len()).unwrap();
            data_payload.extend_from_slice(bytes);

            push_gguf_str(&mut buf, name);
            buf.extend_from_slice(&(u32::try_from(shape.len()).unwrap()).to_le_bytes());
            for &dim in shape {
                buf.extend_from_slice(&(u64::try_from(dim).unwrap()).to_le_bytes());
            }
            buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
            buf.extend_from_slice(&rel_off.to_le_bytes());
        }

        let rem = buf.len() % align;
        if rem != 0 {
            buf.resize(buf.len() + (align - rem), 0);
        }
        buf.extend_from_slice(&data_payload);
        buf
    }

    fn compact_test_recipe() -> WhisperRecipe {
        WhisperRecipe {
            n_mels: 80,
            n_audio_ctx: 8,
            n_audio_state: 16,
            n_audio_head: 2,
            n_audio_layer: 2,
            audio_ffn_dim: 32,
            n_text_ctx: 12,
            n_text_state: 16,
            n_text_head: 2,
            n_text_layer: 2,
            text_ffn_dim: 32,
            n_vocab: 24,
            norm_eps: 1e-5,
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "constructs complete deterministic OpenAI-named Whisper checkpoint for end-to-end tests"
    )]
    fn build_whisper_tensors(
        recipe: &WhisperRecipe,
        hf_names: bool,
        untied_proj: bool,
    ) -> OwnedWeights {
        let mut entries = Vec::new();
        let d_enc = recipe.n_audio_state;
        let d_dec = recipe.n_text_state;
        let mels = recipe.n_mels;

        let mut push = |name: String, quant: QuantType, shape: Vec<usize>, vals: &[f32]| {
            let bytes = match quant {
                QuantType::F16 => encode_f16(vals),
                _ => encode_f32(vals),
            };
            entries.push(OwnedTensor {
                name,
                quant,
                shape,
                bytes,
            });
        };

        let enc_conv1_n = if hf_names {
            "model.encoder.conv1.weight"
        } else {
            "encoder.conv1.weight"
        };
        let enc_conv1_b = if hf_names {
            "model.encoder.conv1.bias"
        } else {
            "encoder.conv1.bias"
        };
        let enc_conv2_n = if hf_names {
            "model.encoder.conv2.weight"
        } else {
            "encoder.conv2.weight"
        };
        let enc_conv2_b = if hf_names {
            "model.encoder.conv2.bias"
        } else {
            "encoder.conv2.bias"
        };
        let enc_pos_n = if hf_names {
            "model.encoder.embed_positions.weight"
        } else {
            "encoder.positional_embedding"
        };
        let enc_ln_weight_name = if hf_names {
            "model.encoder.layer_norm.weight"
        } else {
            "encoder.ln_post.weight"
        };
        let enc_ln_bias_name = if hf_names {
            "model.encoder.layer_norm.bias"
        } else {
            "encoder.ln_post.bias"
        };

        push(
            enc_conv1_n.to_owned(),
            QuantType::F32,
            vec![d_enc, mels, 3],
            &det_vec(d_enc * mels * 3, 1, 0.04),
        );
        push(
            enc_conv1_b.to_owned(),
            QuantType::F32,
            vec![d_enc],
            &det_vec(d_enc, 2, 0.02),
        );
        push(
            enc_conv2_n.to_owned(),
            QuantType::F32,
            vec![d_enc, d_enc, 3],
            &det_vec(d_enc * d_enc * 3, 3, 0.04),
        );
        push(
            enc_conv2_b.to_owned(),
            QuantType::F32,
            vec![d_enc],
            &det_vec(d_enc, 4, 0.02),
        );
        push(
            enc_pos_n.to_owned(),
            QuantType::F32,
            vec![recipe.n_audio_ctx, d_enc],
            &det_vec(recipe.n_audio_ctx * d_enc, 5, 0.03),
        );
        push(
            enc_ln_weight_name.to_owned(),
            QuantType::F32,
            vec![d_enc],
            &det_ones_perturbed(d_enc, 6),
        );
        push(
            enc_ln_bias_name.to_owned(),
            QuantType::F32,
            vec![d_enc],
            &det_vec(d_enc, 7, 0.01),
        );

        for l in 0..recipe.n_audio_layer {
            let s = 20 + l * 30;
            let (aln_w, aln_b, qw, qb, kw, vw, vb, ow, ob, mln_w, mln_b, f1w, f1b, f2w, f2b) =
                if hf_names {
                    (
                        format!("model.encoder.layers.{l}.self_attn_layer_norm.weight"),
                        format!("model.encoder.layers.{l}.self_attn_layer_norm.bias"),
                        format!("model.encoder.layers.{l}.self_attn.q_proj.weight"),
                        format!("model.encoder.layers.{l}.self_attn.q_proj.bias"),
                        format!("model.encoder.layers.{l}.self_attn.k_proj.weight"),
                        format!("model.encoder.layers.{l}.self_attn.v_proj.weight"),
                        format!("model.encoder.layers.{l}.self_attn.v_proj.bias"),
                        format!("model.encoder.layers.{l}.self_attn.out_proj.weight"),
                        format!("model.encoder.layers.{l}.self_attn.out_proj.bias"),
                        format!("model.encoder.layers.{l}.final_layer_norm.weight"),
                        format!("model.encoder.layers.{l}.final_layer_norm.bias"),
                        format!("model.encoder.layers.{l}.fc1.weight"),
                        format!("model.encoder.layers.{l}.fc1.bias"),
                        format!("model.encoder.layers.{l}.fc2.weight"),
                        format!("model.encoder.layers.{l}.fc2.bias"),
                    )
                } else {
                    (
                        format!("encoder.blocks.{l}.attn_ln.weight"),
                        format!("encoder.blocks.{l}.attn_ln.bias"),
                        format!("encoder.blocks.{l}.attn.query.weight"),
                        format!("encoder.blocks.{l}.attn.query.bias"),
                        format!("encoder.blocks.{l}.attn.key.weight"),
                        format!("encoder.blocks.{l}.attn.value.weight"),
                        format!("encoder.blocks.{l}.attn.value.bias"),
                        format!("encoder.blocks.{l}.attn.out.weight"),
                        format!("encoder.blocks.{l}.attn.out.bias"),
                        format!("encoder.blocks.{l}.mlp_ln.weight"),
                        format!("encoder.blocks.{l}.mlp_ln.bias"),
                        format!("encoder.blocks.{l}.mlp.0.weight"),
                        format!("encoder.blocks.{l}.mlp.0.bias"),
                        format!("encoder.blocks.{l}.mlp.2.weight"),
                        format!("encoder.blocks.{l}.mlp.2.bias"),
                    )
                };
            push(
                aln_w,
                QuantType::F32,
                vec![d_enc],
                &det_ones_perturbed(d_enc, s + 1),
            );
            push(
                aln_b,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 2, 0.01),
            );
            push(
                qw,
                QuantType::F32,
                vec![d_enc, d_enc],
                &det_vec(d_enc * d_enc, s + 3, 0.05),
            );
            push(
                qb,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 4, 0.02),
            );
            push(
                kw,
                QuantType::F32,
                vec![d_enc, d_enc],
                &det_vec(d_enc * d_enc, s + 5, 0.05),
            );
            push(
                vw,
                QuantType::F32,
                vec![d_enc, d_enc],
                &det_vec(d_enc * d_enc, s + 6, 0.05),
            );
            push(
                vb,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 7, 0.02),
            );
            push(
                ow,
                QuantType::F32,
                vec![d_enc, d_enc],
                &det_vec(d_enc * d_enc, s + 8, 0.05),
            );
            push(
                ob,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 9, 0.02),
            );
            push(
                mln_w,
                QuantType::F32,
                vec![d_enc],
                &det_ones_perturbed(d_enc, s + 10),
            );
            push(
                mln_b,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 11, 0.01),
            );
            push(
                f1w,
                QuantType::F32,
                vec![recipe.audio_ffn_dim, d_enc],
                &det_vec(recipe.audio_ffn_dim * d_enc, s + 12, 0.04),
            );
            push(
                f1b,
                QuantType::F32,
                vec![recipe.audio_ffn_dim],
                &det_vec(recipe.audio_ffn_dim, s + 13, 0.02),
            );
            push(
                f2w,
                QuantType::F32,
                vec![d_enc, recipe.audio_ffn_dim],
                &det_vec(d_enc * recipe.audio_ffn_dim, s + 14, 0.04),
            );
            push(
                f2b,
                QuantType::F32,
                vec![d_enc],
                &det_vec(d_enc, s + 15, 0.02),
            );
        }

        let dec_tok_n = if hf_names {
            "model.decoder.embed_tokens.weight"
        } else {
            "decoder.token_embedding.weight"
        };
        let dec_pos_n = if hf_names {
            "model.decoder.embed_positions.weight"
        } else {
            "decoder.positional_embedding"
        };
        let dec_ln_weight_name = if hf_names {
            "model.decoder.layer_norm.weight"
        } else {
            "decoder.ln.weight"
        };
        let dec_ln_bias_name = if hf_names {
            "model.decoder.layer_norm.bias"
        } else {
            "decoder.ln.bias"
        };

        push(
            dec_tok_n.to_owned(),
            QuantType::F32,
            vec![recipe.n_vocab, d_dec],
            &det_vec(recipe.n_vocab * d_dec, 101, 0.08),
        );
        push(
            dec_pos_n.to_owned(),
            QuantType::F32,
            vec![recipe.n_text_ctx, d_dec],
            &det_vec(recipe.n_text_ctx * d_dec, 102, 0.03),
        );
        push(
            dec_ln_weight_name.to_owned(),
            QuantType::F32,
            vec![d_dec],
            &det_ones_perturbed(d_dec, 103),
        );
        push(
            dec_ln_bias_name.to_owned(),
            QuantType::F32,
            vec![d_dec],
            &det_vec(d_dec, 104, 0.01),
        );

        if untied_proj {
            push(
                "proj_out.weight".to_owned(),
                QuantType::F32,
                vec![recipe.n_vocab, d_dec],
                &det_vec(recipe.n_vocab * d_dec, 105, 0.09),
            );
        }

        for layer_idx in 0..recipe.n_text_layer {
            let seed_base = 200 + layer_idx * 40;
            let prefix = if hf_names {
                format!("model.decoder.layers.{layer_idx}")
            } else {
                format!("decoder.blocks.{layer_idx}")
            };
            let (aln, q_name, k_name, v_name, o_name, cln, cq, ck, cv, co, mln, f1, f2) =
                if hf_names {
                    (
                        "self_attn_layer_norm",
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.out_proj",
                        "encoder_attn_layer_norm",
                        "encoder_attn.q_proj",
                        "encoder_attn.k_proj",
                        "encoder_attn.v_proj",
                        "encoder_attn.out_proj",
                        "final_layer_norm",
                        "fc1",
                        "fc2",
                    )
                } else {
                    (
                        "attn_ln",
                        "attn.query",
                        "attn.key",
                        "attn.value",
                        "attn.out",
                        "cross_attn_ln",
                        "cross_attn.query",
                        "cross_attn.key",
                        "cross_attn.value",
                        "cross_attn.out",
                        "mlp_ln",
                        "mlp.0",
                        "mlp.2",
                    )
                };

            push(
                format!("{prefix}.{aln}.weight"),
                QuantType::F32,
                vec![d_dec],
                &det_ones_perturbed(d_dec, seed_base + 1),
            );
            push(
                format!("{prefix}.{aln}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 2, 0.01),
            );
            push(
                format!("{prefix}.{q_name}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 3, 0.05),
            );
            push(
                format!("{prefix}.{q_name}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 4, 0.02),
            );
            push(
                format!("{prefix}.{k_name}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 5, 0.05),
            );
            push(
                format!("{prefix}.{v_name}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 6, 0.05),
            );
            push(
                format!("{prefix}.{v_name}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 7, 0.02),
            );
            push(
                format!("{prefix}.{o_name}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 8, 0.05),
            );
            push(
                format!("{prefix}.{o_name}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 9, 0.02),
            );

            push(
                format!("{prefix}.{cln}.weight"),
                QuantType::F32,
                vec![d_dec],
                &det_ones_perturbed(d_dec, seed_base + 10),
            );
            push(
                format!("{prefix}.{cln}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 11, 0.01),
            );
            push(
                format!("{prefix}.{cq}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 12, 0.05),
            );
            push(
                format!("{prefix}.{cq}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 13, 0.02),
            );
            push(
                format!("{prefix}.{ck}.weight"),
                QuantType::F32,
                vec![d_dec, d_enc],
                &det_vec(d_dec * d_enc, seed_base + 14, 0.06),
            );
            push(
                format!("{prefix}.{cv}.weight"),
                QuantType::F32,
                vec![d_dec, d_enc],
                &det_vec(d_dec * d_enc, seed_base + 15, 0.06),
            );
            push(
                format!("{prefix}.{cv}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 16, 0.02),
            );
            push(
                format!("{prefix}.{co}.weight"),
                QuantType::F32,
                vec![d_dec, d_dec],
                &det_vec(d_dec * d_dec, seed_base + 17, 0.05),
            );
            push(
                format!("{prefix}.{co}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 18, 0.02),
            );

            push(
                format!("{prefix}.{mln}.weight"),
                QuantType::F32,
                vec![d_dec],
                &det_ones_perturbed(d_dec, seed_base + 19),
            );
            push(
                format!("{prefix}.{mln}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 20, 0.01),
            );
            push(
                format!("{prefix}.{f1}.weight"),
                QuantType::F32,
                vec![recipe.text_ffn_dim, d_dec],
                &det_vec(recipe.text_ffn_dim * d_dec, seed_base + 21, 0.04),
            );
            push(
                format!("{prefix}.{f1}.bias"),
                QuantType::F32,
                vec![recipe.text_ffn_dim],
                &det_vec(recipe.text_ffn_dim, seed_base + 22, 0.02),
            );
            push(
                format!("{prefix}.{f2}.weight"),
                QuantType::F32,
                vec![d_dec, recipe.text_ffn_dim],
                &det_vec(d_dec * recipe.text_ffn_dim, seed_base + 23, 0.04),
            );
            push(
                format!("{prefix}.{f2}.bias"),
                QuantType::F32,
                vec![d_dec],
                &det_vec(d_dec, seed_base + 24, 0.02),
            );
        }

        OwnedWeights { entries }
    }

    fn greedy_argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        u32::try_from(best_idx).expect("vocab index fits u32")
    }

    fn assert_f32_parity(lhs: &[f32], rhs: &[f32], label: &str) {
        assert_eq!(lhs.len(), rhs.len(), "{label} length mismatch");
        // Miri introduces small bounded ULP perturbations in transcendental ops (sin/cos/exp/log10).
        let tol = if cfg!(miri) { 1e-5 } else { 0.0 };
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "{label} mismatch at {idx}: {a} vs {b} (tol={tol})"
            );
        }
    }

    fn sine_pcm(num_samples: usize, freq_hz: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "sample index fits in f32")]
                let t = (i as f32) / 16_000.0_f32;
                (2.0_f32 * core::f32::consts::PI * freq_hz * t).sin() * 0.5_f32
            })
            .collect()
    }

    #[test]
    fn bootstrap_metadata_matches_policy() {
        assert_eq!(CRATE_NAME, "jero-arch-whisper");
        assert_eq!(
            CRATE_LICENSE,
            "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
        );
        assert_ne!(crate_description(), "");
        assert_eq!(jero_core::DEFAULT_ALIGNMENT, 32);
    }

    #[test]
    fn test_whisper_presets_and_capabilities() {
        let tiny = WhisperRecipe::tiny();
        assert_eq!(tiny.architecture_id(), "whisper");
        assert!(tiny.requires_kv_cache());
        assert_eq!(tiny.n_mels, 80);
        assert_eq!(tiny.n_audio_state, 384);
        assert_eq!(tiny.n_audio_head, 6);
        assert_eq!(tiny.n_audio_layer, 4);
        assert_eq!(tiny.n_text_layer, 4);
        assert_eq!(tiny.audio_head_dim(), 64);

        let base = WhisperRecipe::base();
        assert_eq!(base.n_audio_state, 512);
        assert_eq!(base.n_audio_head, 8);
        assert_eq!(base.n_audio_layer, 6);

        let small = WhisperRecipe::small();
        assert_eq!(small.n_audio_state, 768);
        assert_eq!(small.n_audio_head, 12);
        assert_eq!(small.n_audio_layer, 12);

        let turbo = WhisperRecipe::large_v3_turbo();
        assert_eq!(turbo.n_mels, 128);
        assert_eq!(turbo.n_audio_state, 1280);
        assert_eq!(turbo.n_audio_head, 20);
        assert_eq!(turbo.n_audio_layer, 32);
        assert_eq!(turbo.n_text_layer, 4);
        assert_eq!(turbo.n_vocab, WHISPER_VOCAB_V3);

        assert_eq!(WhisperRecipe::from_variant("tiny").unwrap(), tiny);
        assert_eq!(WhisperRecipe::from_variant("whisper-base").unwrap(), base);
        assert_eq!(WhisperRecipe::from_variant("SMALL").unwrap(), small);
        assert_eq!(
            WhisperRecipe::from_variant("large-v3-turbo").unwrap(),
            turbo
        );
        assert!(WhisperRecipe::from_variant("unknown-model").is_err());

        assert_eq!(WhisperRecipe::encoder_output_len(3000), Some(1500));
        assert_eq!(WhisperRecipe::encoder_output_len(8), Some(4));
        assert_eq!(WhisperRecipe::encoder_output_len(0), None);
        assert_eq!(turbo.cross_attn_kv_len(3000).unwrap(), 4 * 2 * 1500 * 1280);

        assert!(base.supports_quant(QuantType::F32));
        assert!(base.supports_quant(QuantType::F16));
        assert!(base.supports_quant(QuantType::BF16));
        assert!(base.supports_quant(QuantType::Q8_0));
        assert!(base.supports_quant(QuantType::Q4_K_M));
        assert!(!base.supports_quant(QuantType::MXFP4));
        assert!(!base.supports_quant(QuantType::I2_S));
    }

    #[test]
    fn test_gguf_and_safetensors_metadata_hydration() {
        // 1. GGUF hydration for large-v3-turbo
        let gguf_kvs = [
            ("whisper.n_mels", GgufValue::Uint32(128)),
            ("whisper.encoder.embedding_length", GgufValue::Uint32(1280)),
            (
                "whisper.encoder.attention.head_count",
                GgufValue::Uint32(20),
            ),
            ("whisper.encoder.layer_count", GgufValue::Uint32(32)),
            (
                "whisper.encoder.feed_forward_length",
                GgufValue::Uint32(5120),
            ),
            ("whisper.decoder.embedding_length", GgufValue::Uint32(1280)),
            (
                "whisper.decoder.attention.head_count",
                GgufValue::Uint32(20),
            ),
            ("whisper.decoder.layer_count", GgufValue::Uint32(4)),
            (
                "whisper.decoder.feed_forward_length",
                GgufValue::Uint32(5120),
            ),
            ("whisper.vocab_size", GgufValue::Uint32(51_866)),
        ];
        let gguf_bytes = create_synthetic_gguf_v3_with_kvs("whisper", 32, &gguf_kvs, &[]);
        let gguf = GgufFile::parse(&gguf_bytes).unwrap();
        let from_gguf = WhisperRecipe::from_gguf(&gguf).unwrap();
        assert_eq!(from_gguf, WhisperRecipe::large_v3_turbo());

        // 2. SafeTensors hydration with __metadata__
        let st_bytes = create_synthetic_safetensors(
            &[
                ("model_type", "whisper"),
                ("num_mel_bins", "80"),
                ("d_model", "768"),
                ("encoder_attention_heads", "12"),
                ("decoder_attention_heads", "12"),
                ("encoder_layers", "12"),
                ("decoder_layers", "12"),
                ("encoder_ffn_dim", "3072"),
                ("decoder_ffn_dim", "3072"),
                ("max_source_positions", "1500"),
                ("max_target_positions", "448"),
                ("vocab_size", "51865"),
            ],
            &[],
        );
        let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
        let from_st = WhisperRecipe::from_safetensors(&st).unwrap();
        assert_eq!(from_st, WhisperRecipe::small());

        // 3. Reject non-Whisper architectures fail-closed
        let llama_gguf = create_synthetic_gguf_v3_with_kvs("llama", 32, &[], &[]);
        let parsed_llama = GgufFile::parse(&llama_gguf).unwrap();
        assert!(matches!(
            WhisperRecipe::from_gguf(&parsed_llama),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_slaney_mel_filters_and_pcm_to_log_mel_parity() {
        let f80 = build_slaney_mel_filters(80, 201).unwrap();
        assert_eq!(f80.len(), 80 * 201);
        assert!(f80.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(f80.iter().any(|&v| v > 0.0));

        let f128 = build_slaney_mel_filters(128, 257).unwrap();
        assert_eq!(f128.len(), 128 * 257);
        assert!(f128.iter().all(|&v| v.is_finite() && v >= 0.0));

        assert!(build_slaney_mel_filters(64, 201).is_err());
        assert!(build_slaney_mel_filters(80, 128).is_err());

        // 16 kHz sine waves at 440 Hz vs 3000 Hz -> 4 frames (640 samples)
        let pcm_low = sine_pcm(640, 440.0);
        let pcm_high = sine_pcm(640, 3000.0);

        let (mel_low, n_frames) = pcm_to_log_mel(&pcm_low, None, 80).unwrap();
        assert_eq!(n_frames, 4);
        assert_eq!(mel_low.len(), 80 * 4);
        assert!(mel_low.iter().all(|v| v.is_finite()));

        // Whisper clamp range: (max - min) <= 8.0 / 4.0 = 2.0
        let max_v = mel_low.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_v = mel_low.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_v - min_v <= 2.0 + 1e-5);

        // Explicit filters.mel TensorView produces identical output to Slaney builder
        let filter_bytes = encode_f32(&f80);
        let filter_view =
            TensorView::from_bytes("filters.mel", QuantType::F32, &[80, 201], 0, &filter_bytes)
                .unwrap();
        let (mel_from_tensor, frames_tensor) =
            pcm_to_log_mel(&pcm_low, Some(&filter_view), 80).unwrap();
        assert_eq!(frames_tensor, 4);
        assert_f32_parity(&mel_low, &mel_from_tensor, "filters.mel parity");

        // Distinct acoustic tones produce distinct Mel spectrograms
        let (mel_high, _) = pcm_to_log_mel(&pcm_high, None, 80).unwrap();
        let diff: f32 = mel_low
            .iter()
            .zip(mel_high.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1.0,
            "distinct frequencies must yield distinct mel features"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "end-to-end GGUF vs SafeTensors parity, batched vs sequential decode, and acoustic sensitivity"
    )]
    fn test_end_to_end_pcm_mel_encoder_cross_attn_decoder_gguf_safetensors_parity() {
        let recipe = compact_test_recipe();
        let weights = build_whisper_tensors(&recipe, false, false);

        // Serialize the exact same model weights into BOTH SafeTensorsFile and GgufFile!
        let st_specs: Vec<(&str, QuantType, &[usize], &[u8])> = weights
            .entries
            .iter()
            .map(|e| {
                (
                    e.name.as_str(),
                    e.quant,
                    e.shape.as_slice(),
                    e.bytes.as_slice(),
                )
            })
            .collect();
        let st_bytes = create_synthetic_safetensors(
            &[
                ("general.architecture", "whisper"),
                ("num_mel_bins", "80"),
                ("d_model", "16"),
                ("encoder_attention_heads", "2"),
                ("decoder_attention_heads", "2"),
                ("encoder_ffn_dim", "32"),
                ("decoder_ffn_dim", "32"),
                ("max_source_positions", "8"),
                ("max_target_positions", "12"),
                ("vocab_size", "24"),
            ],
            &st_specs,
        );
        let st_file = SafeTensorsFile::from_bytes(&st_bytes).expect("parse SafeTensors");
        let st_recipe = WhisperRecipe::from_safetensors(&st_file).expect("hydrate SafeTensors");
        assert_eq!(st_recipe, recipe);
        st_recipe
            .validate_model(st_file.metadata(), &st_file)
            .expect("validate SafeTensors");

        // GGUF reverses ne[0..ndim] shape order while preserving identical row-major payloads!
        let gguf_shapes: Vec<Vec<usize>> = weights
            .entries
            .iter()
            .map(|e| e.shape.iter().copied().rev().collect())
            .collect();
        let gguf_specs: Vec<(&str, QuantType, &[usize], &[u8])> = weights
            .entries
            .iter()
            .zip(gguf_shapes.iter())
            .map(|(e, s)| (e.name.as_str(), e.quant, s.as_slice(), e.bytes.as_slice()))
            .collect();
        let gguf_kvs = [
            ("whisper.n_mels", 80_u32),
            ("whisper.encoder.embedding_length", 16),
            ("whisper.encoder.attention.head_count", 2),
            ("whisper.encoder.layer_count", 2),
            ("whisper.encoder.feed_forward_length", 32),
            ("whisper.encoder.context_length", 8),
            ("whisper.decoder.embedding_length", 16),
            ("whisper.decoder.attention.head_count", 2),
            ("whisper.decoder.layer_count", 2),
            ("whisper.decoder.feed_forward_length", 32),
            ("whisper.decoder.context_length", 12),
            ("whisper.vocab_size", 24),
        ];
        let gguf_bytes = build_gguf_v3_with_payloads("whisper", &gguf_kvs, &gguf_specs);
        let gguf_file = GgufFile::parse(&gguf_bytes).expect("parse GGUF");
        let gguf_recipe = WhisperRecipe::from_gguf(&gguf_file).expect("hydrate GGUF");
        assert_eq!(gguf_recipe, recipe);
        gguf_recipe
            .validate_model(gguf_file.metadata(), &gguf_file)
            .expect("validate GGUF");

        // 1. PCM -> Mel (4 frames -> conv2 stride 2 -> 2 encoder positions)
        let pcm = sine_pcm(640, 440.0);
        let (mel, n_frames) = st_recipe.pcm_to_log_mel(&pcm, None).unwrap();
        assert_eq!(n_frames, 4);
        let cross_len = st_recipe.cross_attn_kv_len(n_frames).unwrap();
        assert_eq!(cross_len, 2 * 2 * 2 * 16);

        let n_ctx = 8_usize;
        let self_kv_len = st_recipe.self_attn_kv_len(n_ctx).unwrap();

        // 2. SafeTensors prefill([1, 2, 3]) + greedy decode loop
        let prompt = [1_u32, 2, 3];
        let mut st_logits = vec![0.0_f32; recipe.n_vocab];
        let mut st_self_kv = vec![0.0_f32; self_kv_len];
        let mut st_cross_kv = vec![0.0_f32; cross_len];
        let mut st_ctx = ExecutionContext::new(&prompt, n_ctx, &mut st_logits, &mut st_self_kv)
            .unwrap()
            .with_kv_geometry(st_recipe.kv_geometry(n_ctx).unwrap())
            .unwrap()
            .with_audio_mel(&mel, 80, n_frames)
            .unwrap()
            .with_cross_attn_kv(&mut st_cross_kv)
            .unwrap();
        st_recipe.prefill_step(&mut st_ctx, &st_file).unwrap();
        assert_eq!(st_ctx.position(), 3);
        let st_prefill_logits = st_ctx.logits().to_vec();
        let first_gen = greedy_argmax(&st_prefill_logits);

        // Snapshot cross_kv to confirm decode_step reuses it verbatim without mutating
        let cross_kv_snapshot = st_ctx.cross_attn_kv().unwrap().to_vec();
        assert!(cross_kv_snapshot.iter().any(|&v| v.abs() > 1e-6));

        st_ctx.prepare_decode(first_gen).unwrap();
        st_recipe.decode_step(&mut st_ctx, &st_file).unwrap();
        assert_eq!(st_ctx.position(), 4);
        assert_eq!(
            st_ctx.cross_attn_kv().unwrap(),
            cross_kv_snapshot.as_slice()
        );
        let st_decode1_logits = st_ctx.logits().to_vec();
        let second_gen = greedy_argmax(&st_decode1_logits);

        // 3. GGUF prefill + decode must match SafeTensors bit-for-bit / within 1e-6
        let mut gg_logits = vec![0.0_f32; recipe.n_vocab];
        let mut gg_self_kv = vec![0.0_f32; self_kv_len];
        let mut gg_cross_kv = vec![0.0_f32; cross_len];
        let mut gg_ctx = ExecutionContext::new(&prompt, n_ctx, &mut gg_logits, &mut gg_self_kv)
            .unwrap()
            .with_audio_mel(&mel, 80, n_frames)
            .unwrap()
            .with_cross_attn_kv(&mut gg_cross_kv)
            .unwrap();
        gguf_recipe.prefill_step(&mut gg_ctx, &gguf_file).unwrap();
        assert_f32_parity(
            gg_ctx.cross_attn_kv().unwrap(),
            &cross_kv_snapshot,
            "GGUF vs SafeTensors cross_attn_kv",
        );
        assert_f32_parity(
            gg_ctx.logits(),
            &st_prefill_logits,
            "GGUF vs SafeTensors prefill logits",
        );

        gg_ctx.prepare_decode(first_gen).unwrap();
        gguf_recipe.decode_step(&mut gg_ctx, &gguf_file).unwrap();
        assert_f32_parity(
            gg_ctx.logits(),
            &st_decode1_logits,
            "GGUF vs SafeTensors decode1 logits",
        );
        assert_eq!(greedy_argmax(gg_ctx.logits()), second_gen);

        // 4. Batched prefill([1, 2, 3]) == sequential prefill([1]) -> decode(2) -> decode(3)
        let mut seq_logits = vec![0.0_f32; recipe.n_vocab];
        let mut seq_self_kv = vec![0.0_f32; self_kv_len];
        let mut seq_ctx = ExecutionContext::new(&[1_u32], n_ctx, &mut seq_logits, &mut seq_self_kv)
            .unwrap()
            .with_cross_attn_kv_readonly(&cross_kv_snapshot, 2, 2 * 2 * 16)
            .unwrap();
        st_recipe.prefill_step(&mut seq_ctx, &st_file).unwrap();
        seq_ctx.prepare_decode(2).unwrap();
        st_recipe.decode_step(&mut seq_ctx, &st_file).unwrap();
        seq_ctx.prepare_decode(3).unwrap();
        st_recipe.decode_step(&mut seq_ctx, &st_file).unwrap();
        for (a, b) in st_prefill_logits.iter().zip(seq_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "batched vs sequential logit mismatch: {a} vs {b}"
            );
        }

        // 5. Distinct audio input changes cross_attn_kv and output logits
        let pcm_diff = sine_pcm(640, 2600.0);
        let (mel_diff, _) = st_recipe.pcm_to_log_mel(&pcm_diff, None).unwrap();
        let mut logits_diff = vec![0.0_f32; recipe.n_vocab];
        let mut self_kv_diff = vec![0.0_f32; self_kv_len];
        let mut cross_kv_diff = vec![0.0_f32; cross_len];
        let mut ctx_diff =
            ExecutionContext::new(&prompt, n_ctx, &mut logits_diff, &mut self_kv_diff)
                .unwrap()
                .with_audio_mel(&mel_diff, 80, n_frames)
                .unwrap()
                .with_cross_attn_kv(&mut cross_kv_diff)
                .unwrap();
        st_recipe.prefill_step(&mut ctx_diff, &st_file).unwrap();
        let max_logit_delta = st_prefill_logits
            .iter()
            .zip(logits_diff.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_logit_delta > 1e-5, "audio change must affect logits");
    }

    #[test]
    fn test_hf_aliases_f16_and_untied_proj_out() {
        let recipe = compact_test_recipe();
        let mut weights = build_whisper_tensors(&recipe, true, true);
        // Also test F16 weights on cross_attn.key and untied proj_out
        let f16_ck_vals = det_vec(16 * 16, 777, 0.05);
        weights.replace(
            "model.decoder.layers.0.encoder_attn.k_proj.weight",
            QuantType::F16,
            vec![16, 16],
            encode_f16(&f16_ck_vals),
        );

        let st_specs: Vec<(&str, QuantType, &[usize], &[u8])> = weights
            .entries
            .iter()
            .map(|e| {
                (
                    e.name.as_str(),
                    e.quant,
                    e.shape.as_slice(),
                    e.bytes.as_slice(),
                )
            })
            .collect();
        let st_bytes = create_synthetic_safetensors(
            &[
                ("model_type", "whisper"),
                ("encoder_attention_heads", "2"),
                ("decoder_attention_heads", "2"),
                ("max_source_positions", "8"),
                ("max_target_positions", "12"),
            ],
            &st_specs,
        );
        let st_file = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
        let hydrated = WhisperRecipe::from_safetensors(&st_file).unwrap();
        assert_eq!(hydrated, recipe);

        let mel = vec![0.25_f32; 80 * 4];
        let cross_len = hydrated.cross_attn_kv_len(4).unwrap();
        let self_len = hydrated.self_attn_kv_len(8).unwrap();
        let mut logits = vec![0.0_f32; recipe.n_vocab];
        let mut self_kv = vec![0.0_f32; self_len];
        let mut cross_kv = vec![0.0_f32; cross_len];
        let mut ctx = ExecutionContext::new(&[0_u32, 4], 8, &mut logits, &mut self_kv)
            .unwrap()
            .with_audio_mel(&mel, 80, 4)
            .unwrap()
            .with_cross_attn_kv(&mut cross_kv)
            .unwrap();
        hydrated.prefill_step(&mut ctx, &st_file).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_adr045_fail_closed_missing_weights_and_kv_guards() {
        let recipe = compact_test_recipe();
        let mel = vec![0.1_f32; 80 * 4];
        let cross_len = recipe.cross_attn_kv_len(4).unwrap();
        let self_len = recipe.self_attn_kv_len(8).unwrap();

        // 1. Empty weights must fail closed (zero synthetic fallback)
        let empty = OwnedWeights {
            entries: Vec::new(),
        };
        let mut cross_kv = vec![0.0_f32; cross_len];
        assert!(
            recipe
                .encode_audio(&empty, &mel, 80, 4, &mut cross_kv)
                .is_err()
        );

        // 2. Missing critical tensors must fail closed in validate_model and execution
        for missing in [
            "encoder.conv1.weight",
            "encoder.conv1.bias",
            "encoder.positional_embedding",
            "encoder.blocks.0.attn.query.weight",
            "encoder.ln_post.weight",
            "decoder.blocks.0.cross_attn.key.weight",
            "decoder.blocks.0.cross_attn.value.weight",
            "decoder.token_embedding.weight",
            "decoder.positional_embedding",
            "decoder.ln.weight",
        ] {
            let mut w = build_whisper_tensors(&recipe, false, false);
            w.remove(missing);
            assert!(
                !w.entries.is_empty() && recipe.validate_weights_presence(&w).is_err(),
                "validate_weights_presence must reject missing '{missing}'"
            );
            let mut logits = vec![0.0_f32; recipe.n_vocab];
            let mut self_kv = vec![0.0_f32; self_len];
            let mut ckv = vec![0.0_f32; cross_len];
            let mut ctx = ExecutionContext::new(&[1_u32], 8, &mut logits, &mut self_kv)
                .unwrap()
                .with_audio_mel(&mel, 80, 4)
                .unwrap()
                .with_cross_attn_kv(&mut ckv)
                .unwrap();
            assert!(
                recipe.prefill_step(&mut ctx, &w).is_err(),
                "prefill_step must fail closed when '{missing}' is absent"
            );
        }

        // 3. Running decoder without cross_attn_kv or with mismatched length fails closed
        let full_w = build_whisper_tensors(&recipe, false, false);
        let mut logits = vec![0.0_f32; recipe.n_vocab];
        let mut self_kv = vec![0.0_f32; self_len];
        let mut ctx_no_ckv = ExecutionContext::new(&[1_u32], 8, &mut logits, &mut self_kv).unwrap();
        assert!(matches!(
            recipe.prefill_step(&mut ctx_no_ckv, &full_w),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        ctx_no_ckv.prepare_decode(2).unwrap();
        assert!(matches!(
            recipe.decode_step(&mut ctx_no_ckv, &full_w),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Wrong cross_attn_kv buffer length
        let mut bad_ckv = vec![0.0_f32; cross_len - 1];
        assert!(matches!(
            recipe.encode_audio(&full_w, &mel, 80, 4, &mut bad_ckv),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut ctx_bad_ckv = ExecutionContext::new(&[1_u32], 8, &mut logits, &mut self_kv)
            .unwrap()
            .with_cross_attn_kv(&mut bad_ckv)
            .unwrap();
        assert!(matches!(
            recipe.prefill_step(&mut ctx_bad_ckv, &full_w),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // ReadOnly cross_attn_kv combined with audio_mel on prefill_step fails closed
        let ro_ckv = vec![0.0_f32; cross_len];
        let mut ctx_ro_mel = ExecutionContext::new(&[1_u32], 8, &mut logits, &mut self_kv)
            .unwrap()
            .with_audio_mel(&mel, 80, 4)
            .unwrap()
            .with_cross_attn_kv_readonly(&ro_ckv, 2, 2 * 2 * 16)
            .unwrap();
        assert!(matches!(
            recipe.prefill_step(&mut ctx_ro_mel, &full_w),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_adr045_fail_closed_corrupted_nonfinite_inputs_and_oob_tokens() {
        let recipe = compact_test_recipe();
        let mel = vec![0.1_f32; 80 * 4];
        let cross_len = recipe.cross_attn_kv_len(4).unwrap();
        let self_len = recipe.self_attn_kv_len(8).unwrap();
        let full_w = build_whisper_tensors(&recipe, false, false);
        let mut cross_kv = vec![0.0_f32; cross_len];
        let ro_ckv = vec![0.0_f32; cross_len];
        let mut logits = vec![0.0_f32; recipe.n_vocab];
        let mut self_kv = vec![0.0_f32; self_len];

        // Non-finite PCM / Mel / weights and out-of-bounds tokens fail closed
        assert!(recipe.pcm_to_log_mel(&[], None).is_err());
        assert!(recipe.pcm_to_log_mel(&[0.0_f32, f32::NAN], None).is_err());
        let mut bad_mel = mel.clone();
        bad_mel[5] = f32::INFINITY;
        assert!(
            recipe
                .encode_audio(&full_w, &bad_mel, 80, 4, &mut cross_kv)
                .is_err()
        );

        let mut nan_w = build_whisper_tensors(&recipe, false, false);
        let mut nan_bias = vec![0.0_f32; 16];
        nan_bias[0] = f32::NAN;
        nan_w.replace(
            "encoder.conv1.bias",
            QuantType::F32,
            vec![16],
            encode_f32(&nan_bias),
        );
        assert!(
            recipe
                .encode_audio(&nan_w, &mel, 80, 4, &mut cross_kv)
                .is_err()
        );

        // Non-finite F16 weight (0x7E00 NaN) fails closed
        let mut f16_nan_w = build_whisper_tensors(&recipe, false, false);
        let mut bad_f16_bytes = encode_f16(&det_vec(16 * 16, 99, 0.05));
        bad_f16_bytes[0] = 0x00;
        bad_f16_bytes[1] = 0x7E;
        f16_nan_w.replace(
            "decoder.blocks.0.cross_attn.key.weight",
            QuantType::F16,
            vec![16, 16],
            bad_f16_bytes,
        );
        assert!(
            recipe
                .encode_audio(&f16_nan_w, &mel, 80, 4, &mut cross_kv)
                .is_err()
        );

        // Out-of-vocab token fails closed
        let bad_tok = [u32::try_from(recipe.n_vocab).unwrap()];
        let mut ctx_bad_tok = ExecutionContext::new(&bad_tok, 8, &mut logits, &mut self_kv)
            .unwrap()
            .with_cross_attn_kv_readonly(&ro_ckv, 2, 2 * 2 * 16)
            .unwrap();
        assert!(matches!(
            recipe.prefill_step(&mut ctx_bad_tok, &full_w),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
