//! Kernel dispatch and portable scalar fallbacks for Whisper audio, attention, norm, and GEMV ops.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    AttentionMaskConfig, BaselineKernel, VectorKernel, bidirectional_attention_f32,
    conv1d_k3_gelu_f32, dequantize_token_row, gelu_inplace_f32, gqa_causal_attention_prefill_f32,
    layernorm_f32, log_mel_spectrogram_whisper_f32, vec_dot_bf16, vec_dot_f16, vec_dot_iq4_nl,
    vec_dot_q2_k, vec_dot_q3_k, vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1,
    vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

#[cfg(not(feature = "with-kernels"))]
use core::f64::consts::PI;

pub(crate) const fn is_supported_quant(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q8_0
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::IQ4_NL
    )
}

pub(crate) fn bounds_err(
    tensor_name: &str,
    expected_elems: usize,
    actual_elems: usize,
    reason: String,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset: 0,
        byte_len: expected_elems.saturating_mul(size_of::<f32>()),
        buffer_len: actual_elems.saturating_mul(size_of::<f32>()),
        reason,
    }
}

pub(crate) fn validate_finite_slice(name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: idx.saturating_mul(size_of::<f32>()),
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(data),
                reason: format!("non-finite f32 value ({val}) in '{name}' at index {idx}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn checked_mul(name: &str, a: usize, b: usize) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| {
        bounds_err(
            name,
            usize::MAX,
            0,
            format!("{name} dimensions ({a} * {b}) overflowed usize"),
        )
    })
}

#[inline]
pub(crate) fn f16_to_f32_ieee(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        f32::from_bits((sign << 31) | (e << 23) | (m << 13))
    } else if exp == 0x1F {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000 | (mant << 13))
        }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

#[inline]
pub(crate) fn bf16_to_f32_ieee(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn validate_raw_half_finite(
    name: &str,
    raw: &[u8],
    num_elements: usize,
    is_bf16: bool,
) -> Result<(), K3Error> {
    let req_bytes = checked_mul(name, num_elements, 2)?;
    if raw.len() < req_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: req_bytes,
            buffer_len: raw.len(),
            reason: format!("half-precision tensor '{name}' byte slice truncated"),
        });
    }
    let exp_mask: u16 = if is_bf16 { 0x7F80 } else { 0x7C00 };
    for (idx, chunk) in raw[..req_bytes].chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        if (bits & exp_mask) == exp_mask {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: idx.saturating_mul(2),
                byte_len: 2,
                buffer_len: raw.len(),
                reason: format!("non-finite half-precision weight in '{name}' at index {idx}"),
            });
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[inline]
pub(crate) fn axpy_f32(dst: &mut [f32], src: &[f32], alpha: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

pub(crate) fn layernorm(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        layernorm_f32(out, input, weight, bias, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if input.is_empty() || out.len() != input.len() || !eps.is_finite() || eps <= 0.0 {
            return Err(bounds_err(
                "layernorm",
                input.len(),
                out.len(),
                "invalid layernorm dimensions or epsilon".to_owned(),
            ));
        }
        validate_finite_slice("layernorm_input", input)?;
        if let Some(w) = weight {
            if w.len() != input.len() {
                return Err(bounds_err(
                    "layernorm_weight",
                    input.len(),
                    w.len(),
                    "layernorm weight length mismatch".to_owned(),
                ));
            }
            validate_finite_slice("layernorm_weight", w)?;
        }
        if let Some(b) = bias {
            if b.len() != input.len() {
                return Err(bounds_err(
                    "layernorm_bias",
                    input.len(),
                    b.len(),
                    "layernorm bias length mismatch".to_owned(),
                ));
            }
            validate_finite_slice("layernorm_bias", b)?;
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "hidden dimension fits comfortably in f32"
        )]
        let inv_n = 1.0_f32 / (input.len() as f32);
        let mean = input.iter().copied().sum::<f32>() * inv_n;
        let var = input
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            * inv_n;
        let inv_std = 1.0_f32 / (var + eps).sqrt();

        for (i, &x) in input.iter().enumerate() {
            let mut val = (x - mean) * inv_std;
            if let Some(w) = weight {
                val *= w[i];
            }
            if let Some(b) = bias {
                val += b[i];
            }
            out[i] = val;
        }
        Ok(())
    }
}

pub(crate) fn gelu_inplace(values: &mut [f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gelu_inplace_f32(values)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if values.is_empty() {
            return Err(bounds_err(
                "gelu_values",
                1,
                0,
                "gelu_values cannot be empty".to_owned(),
            ));
        }
        validate_finite_slice("gelu_values", values)?;
        let sqrt_2_over_pi = (2.0_f32 / core::f32::consts::PI).sqrt();
        for v in values.iter_mut() {
            let x = *v;
            let inner = sqrt_2_over_pi * (x + 0.044_715_f32 * x * x * x);
            *v = 0.5_f32 * x * (1.0_f32 + inner.tanh());
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Conv1D k=3 + GELU dispatch forwards tensor geometry and slices cleanly"
)]
pub(crate) fn conv1d_k3_gelu(
    out: &mut [f32],
    input: &[f32],
    in_channels: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        conv1d_k3_gelu_f32(
            out,
            input,
            in_channels,
            in_len,
            weight,
            bias,
            out_channels,
            stride,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if in_channels == 0 || in_len == 0 || out_channels == 0 || stride == 0 {
            return Err(bounds_err(
                "conv1d",
                1,
                0,
                "conv1d dimensions and stride must be non-zero".to_owned(),
            ));
        }
        let out_len = (in_len - 1) / stride + 1;
        let in_elems = checked_mul("conv1d_input", in_channels, in_len)?;
        let oc_ic = checked_mul("conv1d_weight", out_channels, in_channels)?;
        let w_elems = checked_mul("conv1d_weight", oc_ic, 3)?;
        let out_elems = checked_mul("conv1d_out", out_channels, out_len)?;
        if input.len() != in_elems
            || weight.len() != w_elems
            || bias.len() != out_channels
            || out.len() != out_elems
        {
            return Err(bounds_err(
                "conv1d",
                out_elems,
                out.len(),
                "conv1d slice lengths mismatch expected dimensions".to_owned(),
            ));
        }
        validate_finite_slice("conv1d_input", input)?;
        validate_finite_slice("conv1d_weight", weight)?;
        validate_finite_slice("conv1d_bias", bias)?;

        for oc in 0..out_channels {
            let out_row = &mut out[oc * out_len..(oc + 1) * out_len];
            let b = bias[oc];
            for (t_out, out_slot) in out_row.iter_mut().enumerate() {
                let center = t_out * stride;
                let mut acc = b;
                for ic in 0..in_channels {
                    let in_row = &input[ic * in_len..(ic + 1) * in_len];
                    let w_base = (oc * in_channels + ic) * 3;
                    let left = if center > 0 { in_row[center - 1] } else { 0.0 };
                    let mid = in_row[center];
                    let right = if center + 1 < in_len {
                        in_row[center + 1]
                    } else {
                        0.0
                    };
                    acc += left * weight[w_base]
                        + mid * weight[w_base + 1]
                        + right * weight[w_base + 2];
                }
                *out_slot = acc;
            }
        }
        gelu_inplace(out)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "bidirectional attention forwards Q/K/V slices and head geometry"
)]
pub(crate) fn bidirectional_attention(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        bidirectional_attention_f32(out, q, k, v, seq_len, n_heads, head_dim, None, scale)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        cross_attention(out, q, k, v, seq_len, seq_len, n_heads, head_dim, scale)
    }
}

/// Evaluates unmasked multi-head cross-attention where `q` (`[q_len, n_heads, head_dim]`)
/// attends to cached encoder `(k_enc, v_enc)` (`[enc_len, n_heads, head_dim]`).
#[expect(
    clippy::too_many_arguments,
    reason = "cross-attention forwards Q, cached encoder K/V slices, and query/encoder sequence geometry"
)]
pub(crate) fn cross_attention(
    out: &mut [f32],
    q: &[f32],
    k_enc: &[f32],
    v_enc: &[f32],
    q_len: usize,
    enc_len: usize,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(), K3Error> {
    if q_len == 0 || enc_len == 0 || n_heads == 0 || head_dim == 0 || !scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "whisper".to_owned(),
            reason: "cross-attention dimensions and scale must be non-zero and finite".to_owned(),
        });
    }
    let hidden = checked_mul("cross_attn_hidden", n_heads, head_dim)?;
    let expected_q = checked_mul("cross_attn_q", q_len, hidden)?;
    let expected_kv = checked_mul("cross_attn_kv", enc_len, hidden)?;
    if out.len() != expected_q
        || q.len() != expected_q
        || k_enc.len() != expected_kv
        || v_enc.len() != expected_kv
    {
        return Err(bounds_err(
            "cross_attn",
            expected_q,
            out.len(),
            "cross-attention buffer length mismatch".to_owned(),
        ));
    }

    let mut scores = vec![0.0_f32; enc_len];
    for m in 0..q_len {
        for h in 0..n_heads {
            let q_off = (m * n_heads + h) * head_dim;
            let q_head = &q[q_off..q_off + head_dim];

            let mut max_score = f32::NEG_INFINITY;
            for (e, slot) in scores.iter_mut().enumerate() {
                let k_off = (e * n_heads + h) * head_dim;
                let s = dot_product_f32(q_head, &k_enc[k_off..k_off + head_dim]) * scale;
                *slot = s;
                if s > max_score {
                    max_score = s;
                }
            }

            softmax_inplace(&mut scores);

            let out_head = &mut out[q_off..q_off + head_dim];
            out_head.fill(0.0);
            for (e, &weight) in scores.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let v_off = (e * n_heads + h) * head_dim;
                axpy_f32(out_head, &v_enc[v_off..v_off + head_dim], weight);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "causal attention forwards Q, K/V planes, sequence positions, and scratch buffer"
)]
pub(crate) fn causal_attention_prefill(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads: usize,
    head_dim: usize,
    start_pos: usize,
    total_seq_len: usize,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_prefill_f32(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads,
            n_heads,
            head_dim,
            total_seq_len,
            AttentionMaskConfig::causal(start_pos, scale),
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if m == 0 || n_heads == 0 || head_dim == 0 || total_seq_len == 0 || !scale.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: "causal attention dimensions and scale must be valid".to_owned(),
            });
        }
        let hidden = checked_mul("causal_hidden", n_heads, head_dim)?;
        let expected_q = checked_mul("causal_q", m, hidden)?;
        let expected_kv = checked_mul("causal_kv", total_seq_len, hidden)?;
        if out.len() != expected_q
            || q.len() != expected_q
            || k_cache.len() < expected_kv
            || v_cache.len() < expected_kv
            || start_pos + m > total_seq_len
            || scores_scratch.len() < start_pos + m
        {
            return Err(bounds_err(
                "causal_attention",
                expected_q,
                out.len(),
                "causal attention buffer length mismatch".to_owned(),
            ));
        }

        for m_idx in 0..m {
            let q_pos = start_pos + m_idx;
            let active_len = (q_pos + 1).min(total_seq_len);
            let scores = &mut scores_scratch[..active_len];
            for h in 0..n_heads {
                let q_off = (m_idx * n_heads + h) * head_dim;
                let q_head = &q[q_off..q_off + head_dim];
                for (s, slot) in scores.iter_mut().enumerate() {
                    let k_off = (s * n_heads + h) * head_dim;
                    *slot = dot_product_f32(q_head, &k_cache[k_off..k_off + head_dim]) * scale;
                }
                softmax_inplace(scores);
                let out_head = &mut out[q_off..q_off + head_dim];
                out_head.fill(0.0);
                for (s, &weight) in scores.iter().enumerate() {
                    if weight == 0.0 {
                        continue;
                    }
                    let v_off = (s * n_heads + h) * head_dim;
                    axpy_f32(out_head, &v_cache[v_off..v_off + head_dim], weight);
                }
            }
        }
        Ok(())
    }
}

fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let w = (*s - max_score).exp();
        *s = w;
        sum += w;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "sequence length fits comfortably in f32"
    )]
    let inv_sum = if sum > 0.0 && sum.is_finite() {
        1.0_f32 / sum
    } else {
        1.0_f32 / (scores.len() as f32)
    };
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

pub(crate) fn log_mel_spectrogram_whisper(
    out_mel: &mut [f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        log_mel_spectrogram_whisper_f32(out_mel, pcm_16khz, mel_filters, n_mels, n_frames)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        log_mel_spectrogram_whisper_scalar_fallback(
            out_mel,
            pcm_16khz,
            mel_filters,
            n_mels,
            n_frames,
        )
    }
}

#[cfg(not(feature = "with-kernels"))]
fn log_mel_spectrogram_whisper_scalar_fallback(
    out_mel: &mut [f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(), K3Error> {
    const N_FFT: usize = 400;
    const FFT_POINTS: usize = 512;
    const HOP: usize = 160;
    const BINS_201: usize = 201;
    const BINS_257: usize = 257;

    if (n_mels != 80 && n_mels != 128) || n_frames == 0 {
        return Err(bounds_err(
            "whisper_mel",
            n_mels,
            n_frames,
            format!("invalid Whisper mel dims (n_mels={n_mels}, n_frames={n_frames})"),
        ));
    }
    let out_elems = checked_mul("whisper_out_mel", n_mels, n_frames)?;
    if out_mel.len() != out_elems {
        return Err(bounds_err(
            "whisper_out_mel",
            out_elems,
            out_mel.len(),
            "output mel buffer length mismatch".to_owned(),
        ));
    }
    let min_pcm = checked_mul("whisper_pcm", n_frames, HOP)?;
    if pcm_16khz.len() < min_pcm {
        return Err(bounds_err(
            "whisper_pcm_16khz",
            min_pcm,
            pcm_16khz.len(),
            "pcm buffer shorter than n_frames * 160".to_owned(),
        ));
    }
    let n_bins = if mel_filters.len() == n_mels * BINS_201 {
        BINS_201
    } else if mel_filters.len() == n_mels * BINS_257 {
        BINS_257
    } else {
        return Err(bounds_err(
            "whisper_mel_filters",
            n_mels * BINS_201,
            mel_filters.len(),
            "mel_filters length must match [n_mels, 201] or [n_mels, 257]".to_owned(),
        ));
    };
    validate_finite_slice("whisper_pcm_16khz", pcm_16khz)?;
    validate_finite_slice("whisper_mel_filters", mel_filters)?;

    let mut hann = [0.0_f32; N_FFT];
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample index fits losslessly in f64"
    )]
    for (n, slot) in hann.iter_mut().enumerate() {
        let theta = 2.0_f64 * PI * (n as f64) / (N_FFT as f64);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Hann value lies in [0.0, 1.0]"
        )]
        {
            *slot = (0.5_f64 * (1.0_f64 - theta.cos())) as f32;
        }
    }

    let mut re = [0.0_f32; FFT_POINTS];
    let mut im = [0.0_f32; FFT_POINTS];
    let mut power = [0.0_f32; BINS_257];
    let mut max_val = f32::NEG_INFINITY;

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP;
        let avail = (pcm_16khz.len() - start).min(N_FFT);
        re.fill(0.0);
        im.fill(0.0);
        for i in 0..avail {
            re[i] = pcm_16khz[start + i] * hann[i];
        }
        fft_512_power_bins(&mut re, &mut im, &mut power[..n_bins]);

        for m in 0..n_mels {
            let row = &mel_filters[m * n_bins..(m + 1) * n_bins];
            let energy = dot_product_f32(row, &power[..n_bins]);
            let log_e = energy.max(1e-10_f32).log10();
            out_mel[m * n_frames + frame_idx] = log_e;
            if log_e > max_val {
                max_val = log_e;
            }
        }
    }

    let min_floor = max_val - 8.0_f32;
    for slot in out_mel.iter_mut() {
        *slot = (slot.max(min_floor) + 4.0_f32) * 0.25_f32;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn fft_512_power_bins(re: &mut [f32; 512], im: &mut [f32; 512], out_bins: &mut [f32]) {
    for i in 0..512_usize {
        #[expect(clippy::cast_possible_truncation, reason = "i < 512 fits in u16")]
        let j = usize::from((i as u16).reverse_bits() >> 7);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    for stage in 1..=9_u32 {
        let m = 1_usize << stage;
        let half = m >> 1;
        #[expect(clippy::cast_precision_loss, reason = "stage width <= 512 fits in f64")]
        let m_f64 = m as f64;
        for j in 0..half {
            #[expect(clippy::cast_precision_loss, reason = "j < 256 fits in f64")]
            let angle = -2.0_f64 * PI * (j as f64) / m_f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "unit-circle twiddle factor fits in f32"
            )]
            let (w_re, w_im) = (angle.cos() as f32, angle.sin() as f32);
            let mut k = 0_usize;
            while k < 512 {
                let i1 = k + j;
                let i2 = i1 + half;
                let (r2, im2) = (re[i2], im[i2]);
                let t_re = w_re * r2 - w_im * im2;
                let t_im = w_re * im2 + w_im * r2;
                let (u_re, u_im) = (re[i1], im[i1]);
                re[i1] = u_re + t_re;
                im[i1] = u_im + t_im;
                re[i2] = u_re - t_re;
                im[i2] = u_im - t_im;
                k += m;
            }
        }
    }
    for (k, slot) in out_bins.iter_mut().enumerate() {
        *slot = re[k] * re[k] + im[k] * im[k];
    }
}

pub(crate) fn matvec_any(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    if out.is_empty() || input.is_empty() {
        return Err(bounds_err(
            weights.name(),
            out.len().max(1),
            0,
            "matvec output and input slices must be non-empty".to_owned(),
        ));
    }
    let shape = weights.shape();
    if shape.len() != 2 {
        return Err(bounds_err(
            weights.name(),
            checked_mul(weights.name(), out.len(), input.len())?,
            weights.num_elements(),
            format!(
                "2D weight '{}' expected shape [{}, {}] or [{}, {}], got {shape:?}",
                weights.name(),
                out.len(),
                input.len(),
                input.len(),
                out.len()
            ),
        ));
    }
    let matches_shape = (shape[0] == out.len() && shape[1] == input.len())
        || (shape[0] == input.len() && shape[1] == out.len());
    if !matches_shape {
        return Err(bounds_err(
            weights.name(),
            checked_mul(weights.name(), out.len(), input.len())?,
            weights.num_elements(),
            format!(
                "weight '{}' shape {shape:?} does not match out({}) x in({})",
                weights.name(),
                out.len(),
                input.len()
            ),
        ));
    }

    match weights.quant() {
        QuantType::F32 => matvec_f32(out, weights, input)?,
        QuantType::F16 => matvec_half(out, weights, input, false)?,
        QuantType::BF16 => matvec_half(out, weights, input, true)?,
        quant if is_supported_quant(quant) => matvec_quantized(out, weights, input)?,
        other => {
            return Err(K3Error::unsupported_tensor_quant(
                "whisper",
                weights.name(),
                other,
            ));
        }
    }

    validate_finite_slice(weights.name(), out)?;
    Ok(())
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    let expected_elems = checked_mul(weights.name(), rows, cols)?;
    let req_bytes = checked_mul(weights.name(), expected_elems, size_of::<f32>())?;
    let raw = weights.as_bytes();
    if raw.len() < req_bytes {
        return Err(bounds_err(
            weights.name(),
            expected_elems,
            raw.len() / size_of::<f32>(),
            format!(
                "F32 tensor '{}' byte length {} < required {req_bytes}",
                weights.name(),
                raw.len()
            ),
        ));
    }

    let mut chunk_buf = [0.0_f32; 256];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let row_bytes = &raw[r * cols * 4..(r + 1) * cols * 4];
        let mut row_sum = 0.0_f32;
        let mut col = 0_usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * 4;
                let val = f32::from_le_bytes([
                    row_bytes[off],
                    row_bytes[off + 1],
                    row_bytes[off + 2],
                    row_bytes[off + 3],
                ]);
                if !val.is_finite() {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: r * cols * 4 + off,
                        byte_len: 4,
                        buffer_len: raw.len(),
                        reason: format!("non-finite F32 weight ({val}) in '{}'", weights.name()),
                    });
                }
                *slot = val;
            }
            row_sum += dot_product_f32(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }
    Ok(())
}

fn matvec_half(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    is_bf16: bool,
) -> Result<(), K3Error> {
    validate_raw_half_finite(
        weights.name(),
        weights.as_bytes(),
        weights.num_elements(),
        is_bf16,
    )?;
    #[cfg(feature = "with-kernels")]
    {
        if is_bf16 {
            vec_dot_bf16(out, weights, input)
        } else {
            vec_dot_f16(out, weights, input)
        }
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let decode = if is_bf16 {
            bf16_to_f32_ieee
        } else {
            f16_to_f32_ieee
        };
        let raw = weights.as_bytes();
        let cols = input.len();
        let mut chunk_buf = [0.0_f32; 256];
        for (r, out_slot) in out.iter_mut().enumerate() {
            let row_bytes = &raw[r * cols * 2..(r + 1) * cols * 2];
            let mut row_sum = 0.0_f32;
            let mut col = 0_usize;
            while col < cols {
                let n = (cols - col).min(chunk_buf.len());
                for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                    let off = (col + i) * 2;
                    *slot = decode(u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]));
                }
                row_sum += dot_product_f32(&chunk_buf[..n], &input[col..col + n]);
                col += n;
            }
            *out_slot = row_sum;
        }
        Ok(())
    }
}

fn matvec_quantized(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        match weights.quant() {
            QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
            QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
            QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
            QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
            QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
            QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
            QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
            QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
            QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
            QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
            QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
            other => Err(K3Error::unsupported_tensor_quant(
                "whisper",
                weights.name(),
                other,
            )),
        }
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let mut row_buf = vec![0.0_f32; input.len()];
        for (row_idx, out_slot) in out.iter_mut().enumerate() {
            dequantize_row_f32(weights, row_idx, &mut row_buf)?;
            *out_slot = dot_product_f32(&row_buf, input);
        }
        Ok(())
    }
}

pub(crate) fn dequantize_row_f32(
    view: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if !is_supported_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "whisper",
            view.name(),
            view.quant(),
        ));
    }
    if out.is_empty() {
        return Err(bounds_err(
            view.name(),
            1,
            0,
            "dequantize_row_f32 output slice must be non-empty".to_owned(),
        ));
    }
    let cols = out.len();
    let total = view.num_elements();
    if total == 0 || total % cols != 0 {
        return Err(bounds_err(
            view.name(),
            cols,
            total,
            format!(
                "tensor '{}' elements {total} not divisible by {cols}",
                view.name()
            ),
        ));
    }
    let num_rows = total / cols;
    if row_idx >= num_rows {
        return Err(bounds_err(
            view.name(),
            num_rows,
            row_idx,
            format!("row_idx {row_idx} out of bounds for {num_rows} rows"),
        ));
    }

    match view.quant() {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            dequantize_dense_row_f32(view, row_idx, cols, out)?;
        }
        _ => {
            #[cfg(feature = "with-kernels")]
            {
                dequantize_token_row(view, row_idx, out)?;
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                return Err(K3Error::unsupported_tensor_quant(
                    "whisper",
                    view.name(),
                    view.quant(),
                ));
            }
        }
    }

    validate_finite_slice(view.name(), out)?;
    Ok(())
}

fn dequantize_dense_row_f32(
    view: &TensorView<'_>,
    row_idx: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let elem_bytes = match view.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::unsupported_tensor_quant(
                "whisper",
                view.name(),
                other,
            ));
        }
    };
    let raw = view.as_bytes();
    let row_bytes = checked_mul(view.name(), cols, elem_bytes)?;
    let start = row_idx * row_bytes;
    let end = start + row_bytes;
    if raw.len() < end {
        return Err(bounds_err(
            view.name(),
            end / elem_bytes,
            raw.len() / elem_bytes,
            format!("{:?} tensor byte slice truncated", view.quant()),
        ));
    }
    match view.quant() {
        QuantType::F32 => {
            for (dst, chunk) in out.iter_mut().zip(raw[start..end].chunks_exact(4)) {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        QuantType::F16 => {
            for (dst, chunk) in out.iter_mut().zip(raw[start..end].chunks_exact(2)) {
                *dst = f16_to_f32_ieee(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        QuantType::BF16 => {
            for (dst, chunk) in out.iter_mut().zip(raw[start..end].chunks_exact(2)) {
                *dst = bf16_to_f32_ieee(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

pub(crate) fn dequantize_all_finite(view: &TensorView<'_>) -> Result<Vec<f32>, K3Error> {
    let total = view.num_elements();
    if total == 0 {
        return Err(bounds_err(
            view.name(),
            1,
            0,
            format!("tensor '{}' cannot be empty", view.name()),
        ));
    }
    if !is_supported_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "whisper",
            view.name(),
            view.quant(),
        ));
    }

    let mut out = vec![0.0_f32; total];
    match view.quant() {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            dequantize_row_f32(view, 0, &mut out)?;
        }
        _ => {
            let row_len = match view.shape() {
                [cols, ..] if *cols > 0 => *cols,
                _ => total,
            };
            let row_count = total / row_len;
            for r in 0..row_count {
                let start = r * row_len;
                dequantize_row_f32(view, r, &mut out[start..start + row_len])?;
            }
        }
    }
    validate_finite_slice(view.name(), &out)?;
    Ok(out)
}
