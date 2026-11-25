//! Whisper 16 kHz audio DSP frontend: Slaney Mel filterbank builder and log-Mel spectrogram.

use jero_core::{K3Error, ModelWeights, TensorView};

use crate::ops::{
    bounds_err, checked_mul, dequantize_all_finite, log_mel_spectrogram_whisper,
    validate_finite_slice,
};

/// Canonical Whisper PCM input sample rate in Hz (`16 kHz`).
pub const WHISPER_SAMPLE_RATE: usize = 16_000;

/// Whisper STFT window length in samples (`25 ms` at `16 kHz`).
pub const WHISPER_N_FFT: usize = 400;

/// Whisper STFT frame hop in samples (`10 ms` at `16 kHz`).
pub const WHISPER_HOP_LENGTH: usize = 160;

/// Whisper 400-sample FFT unique frequency bin count (`400 / 2 + 1`).
pub const WHISPER_N_BINS_201: usize = 201;

/// Full 512-point zero-padded FFT unique frequency bin count (`512 / 2 + 1`).
pub const WHISPER_N_BINS_257: usize = 257;

/// Standard Whisper audio chunk duration in seconds (`30 s`).
pub const WHISPER_CHUNK_SECONDS: usize = 30;

/// Standard 30-second Mel spectrogram frame count (`30 * 16_000 / 160 = 3000`).
pub const WHISPER_MAX_FRAMES: usize = 3_000;

const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1_000.0;
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

#[inline]
fn hz_to_mel_slaney(freq_hz: f64) -> f64 {
    let logstep = 6.4_f64.ln() / 27.0;
    if freq_hz >= SLANEY_MIN_LOG_HZ {
        SLANEY_MIN_LOG_MEL + (freq_hz / SLANEY_MIN_LOG_HZ).ln() / logstep
    } else {
        freq_hz / SLANEY_F_SP
    }
}

#[inline]
fn mel_to_hz_slaney(mel: f64) -> f64 {
    let logstep = 6.4_f64.ln() / 27.0;
    if mel >= SLANEY_MIN_LOG_MEL {
        SLANEY_MIN_LOG_HZ * ((mel - SLANEY_MIN_LOG_MEL) * logstep).exp()
    } else {
        mel * SLANEY_F_SP
    }
}

/// Builds a deterministic Slaney-normalized triangular Mel filterbank matrix of shape
/// `[n_mels, n_bins]` matching `librosa.filters.mel(sr=16000, htk=False, norm='slaney')`.
///
/// Supports `n_mels` in `{80, 128}` and `n_bins` in `{201, 257}`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if `n_mels` is not `80` or `128` or if `n_bins` is
/// not `201` or `257`.
pub fn build_slaney_mel_filters(n_mels: usize, n_bins: usize) -> Result<Vec<f32>, K3Error> {
    if n_mels != 80 && n_mels != 128 {
        return Err(bounds_err(
            "whisper_mel_filters",
            80,
            n_mels,
            format!("Whisper Mel filterbank expects n_mels in {{80, 128}}, got {n_mels}"),
        ));
    }
    if n_bins != WHISPER_N_BINS_201 && n_bins != WHISPER_N_BINS_257 {
        return Err(bounds_err(
            "whisper_mel_filters",
            WHISPER_N_BINS_201,
            n_bins,
            format!("Whisper Mel filterbank expects n_bins in {{201, 257}}, got {n_bins}"),
        ));
    }

    let min_mel = hz_to_mel_slaney(0.0);
    #[expect(
        clippy::cast_precision_loss,
        reason = "8000 Hz Nyquist frequency fits losslessly in f64"
    )]
    let max_mel = hz_to_mel_slaney((WHISPER_SAMPLE_RATE / 2) as f64);

    let mut mel_pts = Vec::with_capacity(n_mels + 2);
    #[expect(
        clippy::cast_precision_loss,
        reason = "n_mels + 1 <= 129 fits losslessly in f64"
    )]
    let denom = (n_mels + 1) as f64;
    for i in 0..=(n_mels + 1) {
        #[expect(clippy::cast_precision_loss, reason = "index <= 129 fits in f64")]
        let m = min_mel + (max_mel - min_mel) * (i as f64) / denom;
        mel_pts.push(mel_to_hz_slaney(m));
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "n_bins - 1 <= 256 fits losslessly in f64"
    )]
    let bin_step_hz = ((WHISPER_SAMPLE_RATE / 2) as f64) / ((n_bins - 1) as f64);
    let mut filters = vec![0.0_f32; n_mels * n_bins];

    for m in 0..n_mels {
        let f_left = mel_pts[m];
        let f_center = mel_pts[m + 1];
        let f_right = mel_pts[m + 2];
        let enorm = 2.0_f64 / (f_right - f_left);
        let row = &mut filters[m * n_bins..(m + 1) * n_bins];

        for (k, slot) in row.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "k < 257 fits losslessly in f64")]
            let freq = (k as f64) * bin_step_hz;
            let lower = (freq - f_left) / (f_center - f_left);
            let upper = (f_right - freq) / (f_right - f_center);
            let val = lower.min(upper).max(0.0_f64) * enorm;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Slaney filter weights lie cleanly within f32 range"
            )]
            {
                *slot = val as f32;
            }
        }
    }

    Ok(filters)
}

pub(crate) fn decode_mel_filter_tensor(
    view: &TensorView<'_>,
    n_mels: usize,
) -> Result<Vec<f32>, K3Error> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(bounds_err(
            view.name(),
            n_mels * WHISPER_N_BINS_201,
            view.num_elements(),
            format!(
                "mel filter tensor '{}' must be 2D, got {shape:?}",
                view.name()
            ),
        ));
    }
    let valid_201 = (shape[0] == n_mels && shape[1] == WHISPER_N_BINS_201)
        || (shape[0] == WHISPER_N_BINS_201 && shape[1] == n_mels);
    let valid_257 = (shape[0] == n_mels && shape[1] == WHISPER_N_BINS_257)
        || (shape[0] == WHISPER_N_BINS_257 && shape[1] == n_mels);
    if !valid_201 && !valid_257 {
        return Err(bounds_err(
            view.name(),
            n_mels * WHISPER_N_BINS_201,
            view.num_elements(),
            format!(
                "mel filter tensor '{}' shape {shape:?} does not match [{n_mels}, 201] or [{n_mels}, 257]",
                view.name()
            ),
        ));
    }
    dequantize_all_finite(view)
}

pub(crate) fn extract_mel_filters_from_weights(
    weights: Option<&dyn ModelWeights>,
    n_mels: usize,
) -> Result<Vec<f32>, K3Error> {
    if let Some(w) = weights {
        for name in ["filters.mel", "mel_filters", "model.mel_filters"] {
            if w.contains_tensor(name) {
                let view = w.tensor(name)?;
                return decode_mel_filter_tensor(&view, n_mels);
            }
        }
    }
    build_slaney_mel_filters(n_mels, WHISPER_N_BINS_201)
}

/// Converts 16 kHz mono `f32` PCM samples into a `[n_mels, n_frames]` Whisper log-Mel spectrogram.
///
/// Uses the optional `filters.mel` [`TensorView`] when supplied (supporting both `[n_mels, 201]`
/// and `[n_mels, 257]` shapes), or generates a deterministic Slaney Mel filterbank via
/// [`build_slaney_mel_filters`] otherwise. Returns `(mel, n_frames)`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if `pcm_16khz` is empty or non-finite, `n_mels` is not
/// `80` or `128`, or `mel_filters` has invalid shape/values.
pub fn pcm_to_log_mel(
    pcm_16khz: &[f32],
    mel_filters: Option<&TensorView<'_>>,
    n_mels: usize,
) -> Result<(Vec<f32>, usize), K3Error> {
    if pcm_16khz.is_empty() {
        return Err(bounds_err(
            "whisper_pcm_16khz",
            WHISPER_HOP_LENGTH,
            0,
            "PCM audio input must not be empty".to_owned(),
        ));
    }
    validate_finite_slice("whisper_pcm_16khz", pcm_16khz)?;

    let filters = match mel_filters {
        Some(view) => decode_mel_filter_tensor(view, n_mels)?,
        None => build_slaney_mel_filters(n_mels, WHISPER_N_BINS_201)?,
    };

    let n_frames = pcm_16khz.len().div_ceil(WHISPER_HOP_LENGTH);
    let mel = pcm_to_log_mel_with_filters(pcm_16khz, Some(&filters), n_mels, n_frames)?;
    Ok((mel, n_frames))
}

/// Converts 16 kHz mono `f32` PCM samples into a fixed-frame `[n_mels, n_frames]` Whisper log-Mel
/// spectrogram using an explicit `mel_filters` slice (or Slaney 201-bin fallback when `None`).
///
/// Zero-pads the trailing hop if `pcm_16khz.len() < n_frames * WHISPER_HOP_LENGTH` while still
/// rejecting empty or non-finite inputs.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on empty/non-finite PCM, zero `n_frames`, invalid
/// `n_mels`, or mismatched `mel_filters`.
pub fn pcm_to_log_mel_with_filters(
    pcm_16khz: &[f32],
    mel_filters: Option<&[f32]>,
    n_mels: usize,
    n_frames: usize,
) -> Result<Vec<f32>, K3Error> {
    if pcm_16khz.is_empty() {
        return Err(bounds_err(
            "whisper_pcm_16khz",
            WHISPER_HOP_LENGTH,
            0,
            "PCM audio input must not be empty".to_owned(),
        ));
    }
    if n_frames == 0 {
        return Err(bounds_err(
            "whisper_n_frames",
            1,
            0,
            "n_frames must be non-zero".to_owned(),
        ));
    }
    validate_finite_slice("whisper_pcm_16khz", pcm_16khz)?;

    let built_filters;
    let filters = if let Some(slice) = mel_filters {
        validate_finite_slice("whisper_mel_filters", slice)?;
        slice
    } else {
        built_filters = build_slaney_mel_filters(n_mels, WHISPER_N_BINS_201)?;
        &built_filters
    };

    let min_pcm = checked_mul("whisper_pcm_16khz", n_frames, WHISPER_HOP_LENGTH)?;
    let out_elems = checked_mul("whisper_out_mel", n_mels, n_frames)?;
    let mut out_mel = vec![0.0_f32; out_elems];

    if pcm_16khz.len() < min_pcm {
        let mut padded = vec![0.0_f32; min_pcm];
        padded[..pcm_16khz.len()].copy_from_slice(pcm_16khz);
        log_mel_spectrogram_whisper(&mut out_mel, &padded, filters, n_mels, n_frames)?;
    } else {
        log_mel_spectrogram_whisper(&mut out_mel, pcm_16khz, filters, n_mels, n_frames)?;
    }

    Ok(out_mel)
}
