//! `WhisperRecipe`: Encoder-decoder speech recognition recipe (`"whisper"`) with 16 kHz
//! log-Mel Conv1d stem, bidirectional audio encoder, cached cross-attention KV, and
//! autoregressive causal text decoder.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, SafeTensorsFile, TensorDescriptor, TensorView,
};

use crate::audio_dsp::{
    WHISPER_HOP_LENGTH, extract_mel_filters_from_weights, pcm_to_log_mel_with_filters,
};
use crate::ops::{
    axpy_f32, bidirectional_attention, bounds_err, causal_attention_prefill, checked_mul,
    conv1d_k3_gelu, cross_attention, dequantize_row_f32, gelu_inplace, is_supported_quant,
    layernorm, matvec_any, validate_finite_slice,
};
use crate::weights::{
    cross_attn_kv_aliases, decoder_block_aliases, decoder_mlp_aliases, encoder_block_aliases,
    encoder_mlp_aliases, find_descriptor, infer_indexed_layer_count, require_1d_f32,
    require_2d_matvec, require_3d_conv1d_f32, require_embedding_table, require_optional_1d_f32,
    resolve_optional_tensor, resolve_tensor,
};

/// Canonical Whisper vocabulary size for multilingual `v1` / `v2` checkpoints.
pub const WHISPER_VOCAB_V2: usize = 51_865;

/// Canonical Whisper vocabulary size for `large-v3` and `large-v3-turbo` checkpoints.
pub const WHISPER_VOCAB_V3: usize = 51_866;

/// Canonical Whisper `<|endoftext|>` token ID for multilingual checkpoints.
pub const WHISPER_EOT_TOKEN: u32 = 50_257;

/// Canonical Whisper `<|startoftranscript|>` token ID for multilingual checkpoints.
pub const WHISPER_SOT_TOKEN: u32 = 50_258;

/// Canonical Whisper `<|transcribe|>` task token ID (`v1`/`v2`).
pub const WHISPER_TRANSCRIBE_TOKEN_V2: u32 = 50_359;

/// Canonical Whisper `<|transcribe|>` task token ID (`large-v3` / `large-v3-turbo`).
pub const WHISPER_TRANSCRIBE_TOKEN_V3: u32 = 50_360;

/// Canonical Whisper `<|notimestamps|>` token ID (`v1`/`v2`).
pub const WHISPER_NO_TIMESTAMPS_TOKEN_V2: u32 = 50_363;

/// Canonical Whisper `<|notimestamps|>` token ID (`large-v3` / `large-v3-turbo`).
pub const WHISPER_NO_TIMESTAMPS_TOKEN_V3: u32 = 50_364;

/// Named `OpenAI` Whisper model size presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhisperVariant {
    /// `whisper-tiny` (`80` mels, `384` `d_model`, `6` heads, `4` encoder + `4` decoder layers).
    Tiny,
    /// `whisper-base` (`80` mels, `512` `d_model`, `8` heads, `6` encoder + `6` decoder layers).
    Base,
    /// `whisper-small` (`80` mels, `768` `d_model`, `12` heads, `12` encoder + `12` decoder layers).
    Small,
    /// `whisper-medium` (`80` mels, `1024` `d_model`, `16` heads, `24` encoder + `24` decoder layers).
    Medium,
    /// `whisper-large-v3-turbo` (`128` mels, `1280` `d_model`, `20` heads, `32` encoder + `4` decoder layers).
    LargeV3Turbo,
}

/// Whisper encoder-decoder automatic speech recognition architecture recipe (`"whisper"`).
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperRecipe {
    /// Number of Mel spectrogram frequency bins (`80` for v1/v2, `128` for large-v3/turbo).
    pub n_mels: usize,
    /// Maximum audio encoder sequence length after stride-2 Conv1d downsampling (default `1500`).
    pub n_audio_ctx: usize,
    /// Audio encoder hidden dimension (`d_model`).
    pub n_audio_state: usize,
    /// Number of audio encoder attention heads.
    pub n_audio_head: usize,
    /// Number of audio encoder transformer blocks.
    pub n_audio_layer: usize,
    /// Audio encoder FFN intermediate width (typically `4 * n_audio_state`).
    pub audio_ffn_dim: usize,
    /// Maximum text decoder context length (`max_target_positions`, default `448`).
    pub n_text_ctx: usize,
    /// Text decoder hidden dimension (`d_model`).
    pub n_text_state: usize,
    /// Number of text decoder self- and cross-attention heads.
    pub n_text_head: usize,
    /// Number of text decoder transformer blocks.
    pub n_text_layer: usize,
    /// Text decoder FFN intermediate width (typically `4 * n_text_state`).
    pub text_ffn_dim: usize,
    /// Vocabulary token count (`51_865` for v1/v2, `51_866` for v3/turbo).
    pub n_vocab: usize,
    /// `LayerNorm` epsilon (default `1e-5`).
    pub norm_eps: f32,
}

impl Default for WhisperRecipe {
    fn default() -> Self {
        Self::base()
    }
}

impl WhisperRecipe {
    /// Creates the default Whisper recipe (`whisper-base`).
    #[must_use]
    pub const fn new() -> Self {
        Self::base()
    }

    /// Creates a `whisper-tiny` configuration (`80` mels, `d_model=384`, `4`/`4` layers).
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            audio_ffn_dim: 1536,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            text_ffn_dim: 1536,
            n_vocab: WHISPER_VOCAB_V2,
            norm_eps: 1e-5,
        }
    }

    /// Creates a `whisper-base` configuration (`80` mels, `d_model=512`, `6`/`6` layers).
    #[must_use]
    pub const fn base() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            audio_ffn_dim: 2048,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
            text_ffn_dim: 2048,
            n_vocab: WHISPER_VOCAB_V2,
            norm_eps: 1e-5,
        }
    }

    /// Creates a `whisper-small` configuration (`80` mels, `d_model=768`, `12`/`12` layers).
    #[must_use]
    pub const fn small() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            audio_ffn_dim: 3072,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            text_ffn_dim: 3072,
            n_vocab: WHISPER_VOCAB_V2,
            norm_eps: 1e-5,
        }
    }

    /// Creates a `whisper-medium` configuration (`80` mels, `d_model=1024`, `24`/`24` layers).
    #[must_use]
    pub const fn medium() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            audio_ffn_dim: 4096,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            text_ffn_dim: 4096,
            n_vocab: WHISPER_VOCAB_V2,
            norm_eps: 1e-5,
        }
    }

    /// Creates a `whisper-large-v3-turbo` configuration (`128` mels, `d_model=1280`, `32`/`4` layers).
    #[must_use]
    pub const fn large_v3_turbo() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            audio_ffn_dim: 5120,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 4,
            text_ffn_dim: 5120,
            n_vocab: WHISPER_VOCAB_V3,
            norm_eps: 1e-5,
        }
    }

    /// Returns the configuration matching a [`WhisperVariant`] preset.
    #[must_use]
    pub const fn for_variant(variant: WhisperVariant) -> Self {
        match variant {
            WhisperVariant::Tiny => Self::tiny(),
            WhisperVariant::Base => Self::base(),
            WhisperVariant::Small => Self::small(),
            WhisperVariant::Medium => Self::medium(),
            WhisperVariant::LargeV3Turbo => Self::large_v3_turbo(),
        }
    }

    /// Resolves a named Whisper model preset (`"tiny"`, `"base"`, `"small"`, `"medium"`,
    /// `"large-v3-turbo"`, with optional `"whisper-"` prefix).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `name` does not match a known Whisper variant.
    pub fn from_variant(name: &str) -> Result<Self, K3Error> {
        let norm = name.trim().to_ascii_lowercase();
        match norm.as_str() {
            "tiny" | "whisper-tiny" | "whisper_tiny" => Ok(Self::tiny()),
            "base" | "whisper-base" | "whisper_base" => Ok(Self::base()),
            "small" | "whisper-small" | "whisper_small" => Ok(Self::small()),
            "medium" | "whisper-medium" | "whisper_medium" => Ok(Self::medium()),
            "large-v3-turbo" | "whisper-large-v3-turbo" | "large_v3_turbo" | "turbo" => {
                Ok(Self::large_v3_turbo())
            }
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: format!("unknown Whisper preset variant '{name}'"),
            }),
        }
    }

    /// Hydrates a [`WhisperRecipe`] from parsed [`ModelMetadata`] (both GGUF and `SafeTensors`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `metadata.architecture()` is non-Whisper or if
    /// inferred dimensions are inconsistent (e.g. `n_audio_state % n_audio_head != 0`).
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, K3Error> {
        validate_whisper_architecture(metadata)?;
        let (n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, audio_ffn_dim) =
            infer_encoder_metadata(metadata);
        let (n_text_ctx, n_text_state, n_text_head, n_text_layer, text_ffn_dim, n_vocab) =
            infer_decoder_metadata(metadata, n_mels, n_audio_state, n_audio_layer);
        let norm_eps = metadata.rms_norm_eps().unwrap_or(1e-5);

        let recipe = Self {
            n_mels,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            audio_ffn_dim,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            text_ffn_dim,
            n_vocab,
            norm_eps,
        };
        recipe.validate_config()?;
        Ok(recipe)
    }

    /// Hydrates a [`WhisperRecipe`] from a parsed [`GgufFile`], applying explicit `whisper.*` GGUF
    /// metadata overrides on top of [`Self::from_metadata`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if metadata is non-Whisper or yields invalid geometry.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(gguf.metadata())?;
        recipe.apply_gguf_encoder_overrides(gguf);
        recipe.apply_gguf_decoder_overrides(gguf);
        if let Some(eps) = gguf
            .metadata_value("whisper.layer_norm_eps")
            .or_else(|| gguf.metadata_value("whisper.attention.layer_norm_epsilon"))
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            recipe.norm_eps = eps;
        }

        recipe.validate_config()?;
        Ok(recipe)
    }

    fn apply_gguf_encoder_overrides(&mut self, gguf: &GgufFile<'_>) {
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.n_mels",
                "whisper.mel_bins",
                "whisper.num_mel_bins",
                "whisper.audio.mel_bins",
            ],
        ) {
            self.n_mels = v;
        }
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.encoder.context_length",
                "whisper.max_source_positions",
                "whisper.audio.context_length",
            ],
        ) {
            self.n_audio_ctx = v;
        }
        let enc_head_keys = &[
            "whisper.encoder.attention.head_count",
            "whisper.audio.attention.head_count",
        ];
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.encoder.embedding_length",
                "whisper.audio.embedding_length",
            ],
        ) {
            self.n_audio_state = v;
            if gguf_usize(gguf, enc_head_keys).is_none()
                && self.n_audio_state % self.n_audio_head != 0
            {
                self.n_audio_head = infer_head_count(self.n_audio_state);
            }
        }
        if let Some(v) = gguf_usize(gguf, enc_head_keys) {
            self.n_audio_head = v;
        }
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.encoder.layer_count",
                "whisper.encoder.block_count",
                "whisper.audio.block_count",
            ],
        ) {
            self.n_audio_layer = v;
        }
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.encoder.feed_forward_length",
                "whisper.audio.feed_forward_length",
            ],
        ) {
            self.audio_ffn_dim = v;
        }
    }

    fn apply_gguf_decoder_overrides(&mut self, gguf: &GgufFile<'_>) {
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.decoder.context_length",
                "whisper.max_target_positions",
                "whisper.text.context_length",
            ],
        ) {
            self.n_text_ctx = v;
        }
        let dec_head_keys = &[
            "whisper.decoder.attention.head_count",
            "whisper.text.attention.head_count",
        ];
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.decoder.embedding_length",
                "whisper.text.embedding_length",
            ],
        ) {
            self.n_text_state = v;
            if gguf_usize(gguf, dec_head_keys).is_none()
                && self.n_text_state % self.n_text_head != 0
            {
                self.n_text_head = infer_head_count(self.n_text_state);
            }
        }
        if let Some(v) = gguf_usize(gguf, dec_head_keys) {
            self.n_text_head = v;
        }
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.decoder.layer_count",
                "whisper.decoder.block_count",
                "whisper.text.block_count",
            ],
        ) {
            self.n_text_layer = v;
        }
        if let Some(v) = gguf_usize(
            gguf,
            &[
                "whisper.decoder.feed_forward_length",
                "whisper.text.feed_forward_length",
            ],
        ) {
            self.text_ffn_dim = v;
        }
        if let Some(v) = gguf_usize(gguf, &["whisper.vocab_size", "whisper.n_vocab"]) {
            self.n_vocab = v;
        }
    }

    /// Hydrates a [`WhisperRecipe`] from a parsed [`SafeTensorsFile`], applying `"__metadata__"`
    /// Hugging Face / `OpenAI` hyperparameter keys on top of [`Self::from_metadata`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if metadata is non-Whisper or yields invalid geometry.
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(st.metadata())?;
        let lookup = |keys: &[&str]| -> Option<usize> {
            for &key in keys {
                if let Some(val) = st
                    .metadata_value(key)
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .filter(|&v| v > 0)
                {
                    return Some(val);
                }
            }
            None
        };

        if let Some(v) = lookup(&["num_mel_bins", "n_mels", "whisper.n_mels"]) {
            recipe.n_mels = v;
        }
        if let Some(v) = lookup(&["max_source_positions", "n_audio_ctx"]) {
            recipe.n_audio_ctx = v;
        }
        if let Some(v) = lookup(&["d_model", "hidden_size", "n_audio_state"]) {
            recipe.n_audio_state = v;
            if lookup(&["encoder_attention_heads", "n_audio_head"]).is_none()
                && recipe.n_audio_state % recipe.n_audio_head != 0
            {
                recipe.n_audio_head = infer_head_count(recipe.n_audio_state);
            }
            if st.metadata_value("n_text_state").is_none() {
                recipe.n_text_state = v;
                if lookup(&["decoder_attention_heads", "n_text_head"]).is_none()
                    && recipe.n_text_state % recipe.n_text_head != 0
                {
                    recipe.n_text_head = infer_head_count(recipe.n_text_state);
                }
            }
        }
        if let Some(v) = lookup(&["n_text_state"]) {
            recipe.n_text_state = v;
            if lookup(&["decoder_attention_heads", "n_text_head"]).is_none()
                && recipe.n_text_state % recipe.n_text_head != 0
            {
                recipe.n_text_head = infer_head_count(recipe.n_text_state);
            }
        }
        if let Some(v) = lookup(&["encoder_attention_heads", "n_audio_head"]) {
            recipe.n_audio_head = v;
        }
        if let Some(v) = lookup(&["decoder_attention_heads", "n_text_head"]) {
            recipe.n_text_head = v;
        }
        if let Some(v) = lookup(&[
            "encoder_layers",
            "n_audio_layer",
            "whisper.encoder.layer_count",
        ]) {
            recipe.n_audio_layer = v;
        }
        if let Some(v) = lookup(&[
            "decoder_layers",
            "n_text_layer",
            "whisper.decoder.layer_count",
        ]) {
            recipe.n_text_layer = v;
        }
        if let Some(v) = lookup(&["encoder_ffn_dim", "audio_ffn_dim"]) {
            recipe.audio_ffn_dim = v;
        }
        if let Some(v) = lookup(&["decoder_ffn_dim", "text_ffn_dim"]) {
            recipe.text_ffn_dim = v;
        }
        if let Some(v) = lookup(&["max_target_positions", "n_text_ctx"]) {
            recipe.n_text_ctx = v;
        }
        if let Some(v) = lookup(&["vocab_size", "n_vocab"]) {
            recipe.n_vocab = v;
        }
        if let Some(eps) = st
            .metadata_value("layer_norm_eps")
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            recipe.norm_eps = eps;
        }

        recipe.validate_config()?;
        Ok(recipe)
    }

    /// Validates all structural Whisper dimensions and `norm_eps` fail-closed.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any dimension is zero, head count does not evenly
    /// divide state dimension, or `norm_eps` is non-finite or `<= 0`.
    pub fn validate_config(&self) -> Result<(), K3Error> {
        let non_zero = [
            ("n_mels", self.n_mels),
            ("n_audio_ctx", self.n_audio_ctx),
            ("n_audio_state", self.n_audio_state),
            ("n_audio_head", self.n_audio_head),
            ("n_audio_layer", self.n_audio_layer),
            ("audio_ffn_dim", self.audio_ffn_dim),
            ("n_text_ctx", self.n_text_ctx),
            ("n_text_state", self.n_text_state),
            ("n_text_head", self.n_text_head),
            ("n_text_layer", self.n_text_layer),
            ("text_ffn_dim", self.text_ffn_dim),
            ("n_vocab", self.n_vocab),
        ];
        for (field, val) in non_zero {
            if val == 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "whisper".to_owned(),
                    reason: format!("Whisper config field '{field}' must be non-zero"),
                });
            }
        }
        if self.n_audio_state % self.n_audio_head != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: format!(
                    "n_audio_state ({}) is not divisible by n_audio_head ({})",
                    self.n_audio_state, self.n_audio_head
                ),
            });
        }
        if self.n_text_state % self.n_text_head != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: format!(
                    "n_text_state ({}) is not divisible by n_text_head ({})",
                    self.n_text_state, self.n_text_head
                ),
            });
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: format!("norm_eps ({}) must be finite and > 0", self.norm_eps),
            });
        }
        Ok(())
    }

    /// Returns the per-head dimension of the audio encoder (`n_audio_state / n_audio_head`).
    #[must_use]
    pub const fn audio_head_dim(&self) -> usize {
        self.n_audio_state / self.n_audio_head
    }

    /// Returns the per-head dimension of the text decoder (`n_text_state / n_text_head`).
    #[must_use]
    pub const fn text_head_dim(&self) -> usize {
        self.n_text_state / self.n_text_head
    }

    /// Returns the encoder sequence length `(n_frames - 1) / 2 + 1` produced after the stride-2
    /// `Conv1d` stem (`None` when `n_frames == 0`).
    #[must_use]
    pub const fn encoder_output_len(n_frames: usize) -> Option<usize> {
        if n_frames == 0 {
            None
        } else {
            Some((n_frames - 1) / 2 + 1)
        }
    }

    /// Returns the required `f32` element count for `cross_attn_kv` (`[n_text_layer, 2, enc_len, n_text_state]`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `n_frames == 0` or multiplication overflows `usize`,
    /// or [`K3Error::CapabilityMismatch`] if recipe dimensions are invalid.
    pub fn cross_attn_kv_len(&self, n_frames: usize) -> Result<usize, K3Error> {
        self.validate_config()?;
        let enc_len = Self::encoder_output_len(n_frames).ok_or_else(|| {
            bounds_err(
                "whisper_n_frames",
                1,
                0,
                "n_frames must be non-zero to size cross_attn_kv".to_owned(),
            )
        })?;
        let kv_per_layer = checked_mul(
            "cross_attn_kv",
            checked_mul("cross_attn_kv", 2, enc_len)?,
            self.n_text_state,
        )?;
        checked_mul("cross_attn_kv", self.n_text_layer, kv_per_layer)
    }

    /// Returns the required `f32` element count for the decoder self-attention KV cache.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] on invalid geometry.
    pub fn self_attn_kv_len(&self, n_ctx: usize) -> Result<usize, K3Error> {
        self.kv_geometry(n_ctx)?.required_f32_len()
    }

    /// Constructs the canonical [`KvGeometry`] for the decoder self-attention KV cache.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on invalid dims.
    pub fn kv_geometry(&self, n_ctx: usize) -> Result<KvGeometry, K3Error> {
        self.validate_config()?;
        KvGeometry::new(
            self.n_text_layer,
            self.n_text_head,
            self.text_head_dim(),
            n_ctx,
            None,
        )
    }

    /// Computes a `[self.n_mels, n_frames]` log-Mel spectrogram from 16 kHz mono `f32` PCM samples.
    ///
    /// Uses `filters.mel` from `weights` when provided, or builds a deterministic Slaney
    /// `80`/`128`-bin filterbank via [`crate::build_slaney_mel_filters`] otherwise.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on empty or non-finite PCM,
    /// [`K3Error::ContextLengthExceeded`] if `enc_len > self.n_audio_ctx`, or
    /// [`K3Error::CapabilityMismatch`] on invalid config.
    pub fn pcm_to_log_mel(
        &self,
        pcm_16khz: &[f32],
        weights: Option<&dyn ModelWeights>,
    ) -> Result<(Vec<f32>, usize), K3Error> {
        self.validate_config()?;
        if pcm_16khz.is_empty() {
            return Err(bounds_err(
                "whisper_pcm_16khz",
                WHISPER_HOP_LENGTH,
                0,
                "PCM audio input must not be empty".to_owned(),
            ));
        }
        validate_finite_slice("whisper_pcm_16khz", pcm_16khz)?;

        let n_frames = pcm_16khz.len().div_ceil(WHISPER_HOP_LENGTH);
        let enc_len = Self::encoder_output_len(n_frames).unwrap_or(0);
        if enc_len > self.n_audio_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: enc_len,
                max_context_tokens: self.n_audio_ctx,
            });
        }

        let filters = extract_mel_filters_from_weights(weights, self.n_mels)?;
        let mel = pcm_to_log_mel_with_filters(pcm_16khz, Some(&filters), self.n_mels, n_frames)?;
        Ok((mel, n_frames))
    }

    /// Executes the full Whisper audio encoder (`conv1` -> `conv2` -> positional embedding ->
    /// `0..n_audio_layer` bidirectional encoder blocks -> `ln_post`) and caches per-decoder-layer
    /// cross-attention `(K_enc, V_enc)` projections into `cross_attn_kv`.
    ///
    /// Output layout in `cross_attn_kv` is `[n_text_layer, 2, enc_len, n_text_state]` where
    /// `enc_len = (n_frames - 1) / 2 + 1`.
    ///
    /// # Errors
    /// Returns [`K3Error`] on invalid config/dimensions, non-finite inputs or weights, missing
    /// tensors, context overflow (`enc_len > n_audio_ctx`), or mismatched `cross_attn_kv` length.
    pub fn encode_audio(
        &self,
        weights: &dyn ModelWeights,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        cross_attn_kv: &mut [f32],
    ) -> Result<(), K3Error> {
        self.validate_config()?;
        if n_mels != self.n_mels || n_frames == 0 {
            return Err(bounds_err(
                "audio_mel",
                self.n_mels,
                n_mels,
                format!(
                    "expected n_mels={} and n_frames > 0, got n_mels={n_mels}, n_frames={n_frames}",
                    self.n_mels
                ),
            ));
        }
        let expected_mel_len = checked_mul("audio_mel", n_mels, n_frames)?;
        if mel.len() != expected_mel_len {
            return Err(bounds_err(
                "audio_mel",
                expected_mel_len,
                mel.len(),
                format!(
                    "audio_mel slice length {} != n_mels({n_mels}) * n_frames({n_frames})",
                    mel.len()
                ),
            ));
        }
        validate_finite_slice("audio_mel", mel)?;

        let enc_len = (n_frames - 1) / 2 + 1;
        if enc_len > self.n_audio_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: enc_len,
                max_context_tokens: self.n_audio_ctx,
            });
        }
        let expected_kv_len = self.cross_attn_kv_len(n_frames)?;
        if cross_attn_kv.len() != expected_kv_len {
            return Err(bounds_err(
                "cross_attn_kv",
                expected_kv_len,
                cross_attn_kv.len(),
                format!(
                    "cross_attn_kv length {} != required {expected_kv_len} ([{}, 2, {enc_len}, {}])",
                    cross_attn_kv.len(),
                    self.n_text_layer,
                    self.n_text_state
                ),
            ));
        }

        let mut hidden = self.run_encoder_stem_and_pos(weights, mel, n_frames, enc_len)?;
        for layer in 0..self.n_audio_layer {
            self.run_encoder_layer(weights, layer, enc_len, &mut hidden)?;
        }

        let ln_post_w = require_1d_f32(
            weights,
            &["encoder.ln_post.weight", "model.encoder.layer_norm.weight"],
            self.n_audio_state,
        )?;
        let ln_post_b = require_1d_f32(
            weights,
            &["encoder.ln_post.bias", "model.encoder.layer_norm.bias"],
            self.n_audio_state,
        )?;
        apply_layernorm_rows(
            &mut hidden,
            enc_len,
            self.n_audio_state,
            &ln_post_w,
            &ln_post_b,
            self.norm_eps,
        )?;

        self.cache_cross_attention_kv(weights, &hidden, enc_len, cross_attn_kv)?;
        validate_finite_slice("cross_attn_kv", cross_attn_kv)?;
        Ok(())
    }

    fn run_encoder_stem_and_pos(
        &self,
        weights: &dyn ModelWeights,
        mel: &[f32],
        n_frames: usize,
        enc_len: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let d_enc = self.n_audio_state;
        let conv1_w = require_3d_conv1d_f32(
            weights,
            &["encoder.conv1.weight", "model.encoder.conv1.weight"],
            d_enc,
            self.n_mels,
        )?;
        let conv1_b = require_1d_f32(
            weights,
            &["encoder.conv1.bias", "model.encoder.conv1.bias"],
            d_enc,
        )?;
        let conv2_w = require_3d_conv1d_f32(
            weights,
            &["encoder.conv2.weight", "model.encoder.conv2.weight"],
            d_enc,
            d_enc,
        )?;
        let conv2_b = require_1d_f32(
            weights,
            &["encoder.conv2.bias", "model.encoder.conv2.bias"],
            d_enc,
        )?;

        let mut conv1_out = vec![0.0_f32; checked_mul("conv1_out", d_enc, n_frames)?];
        conv1d_k3_gelu(
            &mut conv1_out,
            mel,
            self.n_mels,
            n_frames,
            &conv1_w,
            &conv1_b,
            d_enc,
            1,
        )?;

        let mut conv2_out = vec![0.0_f32; checked_mul("conv2_out", d_enc, enc_len)?];
        conv1d_k3_gelu(
            &mut conv2_out,
            &conv1_out,
            d_enc,
            n_frames,
            &conv2_w,
            &conv2_b,
            d_enc,
            2,
        )?;

        let (pos_view, _) = require_embedding_table(
            weights,
            &[
                "encoder.positional_embedding",
                "encoder.positional_embedding.weight",
                "model.encoder.embed_positions.weight",
            ],
            enc_len,
            d_enc,
        )?;

        let mut hidden = vec![0.0_f32; conv2_out.len()];
        let mut pos_row = vec![0.0_f32; d_enc];
        for t in 0..enc_len {
            dequantize_row_f32(&pos_view, t, &mut pos_row)?;
            let dst = &mut hidden[t * d_enc..(t + 1) * d_enc];
            for (d, slot) in dst.iter_mut().enumerate() {
                *slot = conv2_out[d * enc_len + t] + pos_row[d];
            }
        }
        Ok(hidden)
    }

    fn run_encoder_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        enc_len: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        self.run_encoder_attn(weights, layer, enc_len, hidden)?;
        self.run_encoder_mlp(weights, layer, enc_len, hidden)
    }

    fn run_encoder_attn(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        enc_len: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.n_audio_state;
        let total = enc_len * d;

        let attn_ln_w = load_enc_1d(
            weights,
            layer,
            "attn_ln.weight",
            "self_attn_layer_norm.weight",
            d,
        )?;
        let attn_ln_b = load_enc_1d(
            weights,
            layer,
            "attn_ln.bias",
            "self_attn_layer_norm.bias",
            d,
        )?;
        let mut norm = hidden.to_vec();
        apply_layernorm_rows(&mut norm, enc_len, d, &attn_ln_w, &attn_ln_b, self.norm_eps)?;

        let q_w = load_enc_2d(
            weights,
            layer,
            "attn.query.weight",
            "self_attn.q_proj.weight",
            d,
            d,
        )?;
        let q_b = load_enc_1d(
            weights,
            layer,
            "attn.query.bias",
            "self_attn.q_proj.bias",
            d,
        )?;
        let k_w = load_enc_2d(
            weights,
            layer,
            "attn.key.weight",
            "self_attn.k_proj.weight",
            d,
            d,
        )?;
        let k_b = load_enc_opt_1d(weights, layer, "attn.key.bias", "self_attn.k_proj.bias", d)?;
        let v_w = load_enc_2d(
            weights,
            layer,
            "attn.value.weight",
            "self_attn.v_proj.weight",
            d,
            d,
        )?;
        let v_b = load_enc_1d(
            weights,
            layer,
            "attn.value.bias",
            "self_attn.v_proj.bias",
            d,
        )?;
        let o_w = load_enc_2d(
            weights,
            layer,
            "attn.out.weight",
            "self_attn.out_proj.weight",
            d,
            d,
        )?;
        let o_b = load_enc_1d(
            weights,
            layer,
            "attn.out.bias",
            "self_attn.out_proj.bias",
            d,
        )?;

        let mut q = vec![0.0_f32; total];
        let mut k = vec![0.0_f32; total];
        let mut v = vec![0.0_f32; total];
        project_rows(&mut q, &norm, enc_len, d, &q_w, Some(&q_b))?;
        project_rows(&mut k, &norm, enc_len, d, &k_w, k_b.as_deref())?;
        project_rows(&mut v, &norm, enc_len, d, &v_w, Some(&v_b))?;

        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let scale = 1.0_f32 / (self.audio_head_dim() as f32).sqrt();
        let mut attn_ctx = vec![0.0_f32; total];
        bidirectional_attention(
            &mut attn_ctx,
            &q,
            &k,
            &v,
            enc_len,
            self.n_audio_head,
            self.audio_head_dim(),
            scale,
        )?;

        let mut attn_out = vec![0.0_f32; total];
        project_rows(&mut attn_out, &attn_ctx, enc_len, d, &o_w, Some(&o_b))?;
        axpy_f32(hidden, &attn_out, 1.0);
        Ok(())
    }

    fn run_encoder_mlp(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        enc_len: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.n_audio_state;
        let ffn_dim = self.audio_ffn_dim;

        let mlp_ln_w = load_enc_1d(
            weights,
            layer,
            "mlp_ln.weight",
            "final_layer_norm.weight",
            d,
        )?;
        let mlp_ln_b = load_enc_1d(weights, layer, "mlp_ln.bias", "final_layer_norm.bias", d)?;
        let mut norm = hidden.to_vec();
        apply_layernorm_rows(&mut norm, enc_len, d, &mlp_ln_w, &mlp_ln_b, self.norm_eps)?;

        let fc1_w = load_enc_mlp_2d(weights, layer, "0.weight", "fc1.weight", ffn_dim, d)?;
        let fc1_b = load_enc_mlp_1d(weights, layer, "0.bias", "fc1.bias", ffn_dim)?;
        let fc2_w = load_enc_mlp_2d(weights, layer, "2.weight", "fc2.weight", d, ffn_dim)?;
        let fc2_b = load_enc_mlp_1d(weights, layer, "2.bias", "fc2.bias", d)?;

        let mut ffn = vec![0.0_f32; enc_len * ffn_dim];
        for t in 0..enc_len {
            let x_row = &norm[t * d..(t + 1) * d];
            let ffn_row = &mut ffn[t * ffn_dim..(t + 1) * ffn_dim];
            matvec_any(ffn_row, &fc1_w, x_row)?;
            axpy_f32(ffn_row, &fc1_b, 1.0);
        }
        gelu_inplace(&mut ffn)?;

        let mut ffn_out = vec![0.0_f32; d];
        for t in 0..enc_len {
            let ffn_row = &ffn[t * ffn_dim..(t + 1) * ffn_dim];
            matvec_any(&mut ffn_out, &fc2_w, ffn_row)?;
            axpy_f32(&mut ffn_out, &fc2_b, 1.0);
            axpy_f32(&mut hidden[t * d..(t + 1) * d], &ffn_out, 1.0);
        }
        Ok(())
    }

    fn cache_cross_attention_kv(
        &self,
        weights: &dyn ModelWeights,
        enc_hidden: &[f32],
        enc_len: usize,
        cross_attn_kv: &mut [f32],
    ) -> Result<(), K3Error> {
        let d_enc = self.n_audio_state;
        let d_dec = self.n_text_state;
        let plane_len = enc_len * d_dec;
        let layer_span = 2 * plane_len;

        for layer in 0..self.n_text_layer {
            let k_aliases = cross_attn_kv_aliases(layer, "key", "k_proj");
            let v_aliases = cross_attn_kv_aliases(layer, "value", "v_proj");
            let k_refs = [
                k_aliases[0].as_str(),
                k_aliases[1].as_str(),
                k_aliases[2].as_str(),
            ];
            let v_refs = [
                v_aliases[0].as_str(),
                v_aliases[1].as_str(),
                v_aliases[2].as_str(),
            ];

            let k_w = require_2d_matvec(weights, &k_refs, d_dec, d_enc)?;
            let k_b = load_dec_opt_1d(
                weights,
                layer,
                "cross_attn.key.bias",
                "encoder_attn.k_proj.bias",
                d_dec,
            )?;
            let v_w = require_2d_matvec(weights, &v_refs, d_dec, d_enc)?;
            let v_b = load_dec_1d(
                weights,
                layer,
                "cross_attn.value.bias",
                "encoder_attn.v_proj.bias",
                d_dec,
            )?;

            let layer_slice = &mut cross_attn_kv[layer * layer_span..(layer + 1) * layer_span];
            let (k_plane, v_plane) = layer_slice.split_at_mut(plane_len);

            for t in 0..enc_len {
                let enc_row = &enc_hidden[t * d_enc..(t + 1) * d_enc];
                let k_dst = &mut k_plane[t * d_dec..(t + 1) * d_dec];
                matvec_any(k_dst, &k_w, enc_row)?;
                if let Some(kb) = k_b.as_deref() {
                    axpy_f32(k_dst, kb, 1.0);
                }

                let v_dst = &mut v_plane[t * d_dec..(t + 1) * d_dec];
                matvec_any(v_dst, &v_w, enc_row)?;
                axpy_f32(v_dst, &v_b, 1.0);
            }
        }
        Ok(())
    }

    fn validate_cross_attn_kv(
        &self,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(Vec<f32>, usize), K3Error> {
        let kv = ctx.cross_attn_kv().ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "whisper".to_owned(),
            reason: "Whisper decoder requires populated cross-attention K/V state via ExecutionContext::with_cross_attn_kv".to_owned(),
        })?;
        let plane_per_layer = checked_mul("cross_attn_kv", 2, self.n_text_state)?;
        let step_stride = checked_mul("cross_attn_kv", self.n_text_layer, plane_per_layer)?;
        if kv.is_empty() || kv.len() % step_stride != 0 {
            return Err(bounds_err(
                "cross_attn_kv",
                step_stride,
                kv.len(),
                format!(
                    "cross_attn_kv length {} must be a positive multiple of n_text_layer({}) * 2 * n_text_state({}) = {step_stride}",
                    kv.len(),
                    self.n_text_layer,
                    self.n_text_state
                ),
            ));
        }
        let enc_len = kv.len() / step_stride;
        if enc_len > self.n_audio_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: enc_len,
                max_context_tokens: self.n_audio_ctx,
            });
        }
        if let Some((n_mels, n_frames)) = ctx.audio_dims() {
            let expected_enc = Self::encoder_output_len(n_frames).unwrap_or(0);
            if n_mels != self.n_mels || expected_enc != enc_len {
                return Err(bounds_err(
                    "cross_attn_kv",
                    expected_enc * step_stride,
                    kv.len(),
                    format!(
                        "cross_attn_kv enc_len={enc_len} mismatches attached audio_mel ({n_mels}x{n_frames} -> {expected_enc} frames)"
                    ),
                ));
            }
        }
        if let Some((d0, d1)) = ctx.cross_attn_kv_dims() {
            let matches_frames_major = d0 == enc_len && d1 == step_stride;
            let matches_state_major =
                d0 == 2 * self.n_text_layer * enc_len && d1 == self.n_text_state;
            if !matches_frames_major && !matches_state_major {
                return Err(bounds_err(
                    "cross_attn_kv",
                    enc_len * step_stride,
                    kv.len(),
                    format!(
                        "cross_attn_kv_dims ({d0}, {d1}) mismatch [n_text_layer={}, 2, enc_len={enc_len}, n_text_state={}]",
                        self.n_text_layer, self.n_text_state
                    ),
                ));
            }
        }
        validate_finite_slice("cross_attn_kv", kv)?;
        Ok((kv.to_vec(), enc_len))
    }

    fn decode_tokens(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), K3Error> {
        let seq_len = tokens.len();
        let expected_geom = self.validate_decode_request(ctx, tokens, start_pos)?;
        let (cross_kv, enc_len) = self.validate_cross_attn_kv(ctx)?;
        let (mut hidden, tok_emb_view) = self.embed_decoder_tokens(weights, tokens, start_pos)?;

        {
            let mut kv_view = KvCacheView::new(ctx.kv_cache_mut(), expected_geom)?;
            for layer in 0..self.n_text_layer {
                self.run_decoder_layer(
                    weights,
                    layer,
                    start_pos,
                    seq_len,
                    enc_len,
                    &cross_kv,
                    &mut kv_view,
                    &mut hidden,
                )?;
            }
        }

        ctx.advance_position(seq_len)?;

        if ctx.compute_logits() {
            let d_dec = self.n_text_state;
            let last_hidden = &hidden[(seq_len - 1) * d_dec..seq_len * d_dec];
            self.project_decoder_logits(ctx, weights, &tok_emb_view, last_hidden)?;
        }

        Ok(())
    }

    fn validate_decode_request(
        &self,
        ctx: &ExecutionContext<'_>,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<KvGeometry, K3Error> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(bounds_err(
                "prompt_tokens",
                1,
                0,
                "decoder requires at least one token".to_owned(),
            ));
        }
        let end_pos = start_pos
            .checked_add(seq_len)
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: self.n_text_ctx,
            })?;
        if end_pos > self.n_text_ctx || end_pos > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: self.n_text_ctx.min(ctx.n_ctx()),
            });
        }
        for (idx, &tok) in tokens.iter().enumerate() {
            if usize::try_from(tok).unwrap_or(usize::MAX) >= self.n_vocab {
                return Err(bounds_err(
                    "decoder.token_embedding.weight",
                    self.n_vocab,
                    tok as usize,
                    format!(
                        "token ID {tok} at index {idx} exceeds n_vocab={}",
                        self.n_vocab
                    ),
                ));
            }
        }
        if ctx.compute_logits() && ctx.logits().len() < self.n_vocab {
            return Err(bounds_err(
                "logits",
                self.n_vocab,
                ctx.logits().len(),
                format!(
                    "logits buffer length {} < n_vocab({})",
                    ctx.logits().len(),
                    self.n_vocab
                ),
            ));
        }

        let expected_geom = self.kv_geometry(ctx.n_ctx())?;
        if let Some(geom) = ctx.kv_geometry()
            && (geom.num_layers != expected_geom.num_layers
                || geom.n_heads_kv != expected_geom.n_heads_kv
                || geom.head_dim != expected_geom.head_dim
                || geom.sliding_window.is_some())
        {
            return Err(bounds_err(
                "kv_cache",
                expected_geom.required_f32_len()?,
                ctx.kv_cache().len(),
                format!(
                    "attached KvGeometry {geom:?} does not match Whisper decoder geometry {expected_geom:?}"
                ),
            ));
        }
        let required_kv = expected_geom.required_f32_len()?;
        if ctx.kv_cache().len() < required_kv {
            return Err(bounds_err(
                "kv_cache",
                required_kv,
                ctx.kv_cache().len(),
                format!(
                    "kv_cache length {} < required {required_kv}",
                    ctx.kv_cache().len()
                ),
            ));
        }
        Ok(expected_geom)
    }

    fn embed_decoder_tokens<'w>(
        &self,
        weights: &'w dyn ModelWeights,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<f32>, TensorView<'w>), K3Error> {
        let seq_len = tokens.len();
        let end_pos = start_pos + seq_len;
        let d_dec = self.n_text_state;
        let (tok_emb_view, _) = require_embedding_table(
            weights,
            &[
                "decoder.token_embedding.weight",
                "decoder.token_embedding",
                "model.decoder.embed_tokens.weight",
                "token_embd.weight",
            ],
            self.n_vocab,
            d_dec,
        )?;
        let (dec_pos_view, _) = require_embedding_table(
            weights,
            &[
                "decoder.positional_embedding",
                "decoder.positional_embedding.weight",
                "model.decoder.embed_positions.weight",
            ],
            end_pos,
            d_dec,
        )?;

        let mut hidden = vec![0.0_f32; checked_mul("decoder_hidden", seq_len, d_dec)?];
        let mut tok_buf = vec![0.0_f32; d_dec];
        let mut pos_buf = vec![0.0_f32; d_dec];
        for (i, &tok) in tokens.iter().enumerate() {
            dequantize_row_f32(&tok_emb_view, tok as usize, &mut tok_buf)?;
            dequantize_row_f32(&dec_pos_view, start_pos + i, &mut pos_buf)?;
            let dst = &mut hidden[i * d_dec..(i + 1) * d_dec];
            for d in 0..d_dec {
                dst[d] = tok_buf[d] + pos_buf[d];
            }
        }
        Ok((hidden, tok_emb_view))
    }

    fn project_decoder_logits(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        tok_emb_view: &TensorView<'_>,
        last_hidden: &[f32],
    ) -> Result<(), K3Error> {
        let d_dec = self.n_text_state;
        let dec_ln_w = require_1d_f32(
            weights,
            &[
                "decoder.ln.weight",
                "model.decoder.layer_norm.weight",
                "decoder.layer_norm.weight",
            ],
            d_dec,
        )?;
        let dec_ln_b = require_1d_f32(
            weights,
            &[
                "decoder.ln.bias",
                "model.decoder.layer_norm.bias",
                "decoder.layer_norm.bias",
            ],
            d_dec,
        )?;
        let mut final_norm = vec![0.0_f32; d_dec];
        layernorm(
            &mut final_norm,
            last_hidden,
            Some(&dec_ln_w),
            Some(&dec_ln_b),
            self.norm_eps,
        )?;

        let untied_aliases = [
            "proj_out.weight",
            "proj_out",
            "decoder.proj_out.weight",
            "lm_head.weight",
            "output.weight",
        ];
        let untied_proj = match resolve_optional_tensor(weights, &untied_aliases)? {
            Some(_) => Some(require_2d_matvec(
                weights,
                &untied_aliases,
                self.n_vocab,
                d_dec,
            )?),
            None => None,
        };
        let proj_view = untied_proj.as_ref().unwrap_or(tok_emb_view);
        let proj_bias = require_optional_1d_f32(
            weights,
            &["proj_out.bias", "decoder.proj_out.bias", "lm_head.bias"],
            self.n_vocab,
        )?;

        let out_logits = &mut ctx.logits_mut()[..self.n_vocab];
        matvec_any(out_logits, proj_view, &final_norm)?;
        if let Some(bias) = proj_bias.as_deref() {
            axpy_f32(out_logits, bias, 1.0);
        }
        validate_finite_slice("whisper_logits", out_logits)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "decoder block requires layer index, positions, cached encoder KV, self KV view, and hidden buffer"
    )]
    fn run_decoder_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        enc_len: usize,
        cross_kv: &[f32],
        kv_view: &mut KvCacheView<'_>,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        self.run_decoder_self_attn(weights, layer, start_pos, seq_len, kv_view, hidden)?;
        self.run_decoder_cross_attn(weights, layer, seq_len, enc_len, cross_kv, hidden)?;
        self.run_decoder_mlp(weights, layer, seq_len, hidden)
    }

    fn run_decoder_self_attn(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        kv_view: &mut KvCacheView<'_>,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.n_text_state;
        let head_dim = self.text_head_dim();
        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let total = seq_len * d;
        let end_pos = start_pos + seq_len;

        let attn_ln_w = load_dec_1d(
            weights,
            layer,
            "attn_ln.weight",
            "self_attn_layer_norm.weight",
            d,
        )?;
        let attn_ln_b = load_dec_1d(
            weights,
            layer,
            "attn_ln.bias",
            "self_attn_layer_norm.bias",
            d,
        )?;
        let mut norm = hidden.to_vec();
        apply_layernorm_rows(&mut norm, seq_len, d, &attn_ln_w, &attn_ln_b, self.norm_eps)?;

        let q_self = self.project_and_cache_decoder_self_qkv(
            weights, layer, start_pos, seq_len, &norm, kv_view,
        )?;

        let o_w = load_dec_2d(
            weights,
            layer,
            "attn.out.weight",
            "self_attn.out_proj.weight",
            d,
            d,
        )?;
        let o_b = load_dec_1d(
            weights,
            layer,
            "attn.out.bias",
            "self_attn.out_proj.bias",
            d,
        )?;

        let mut self_ctx = vec![0.0_f32; total];
        let mut scores_scratch = vec![0.0_f32; end_pos];
        let (k_plane, v_plane) = kv_view.valid_kv_range(layer, end_pos)?;
        causal_attention_prefill(
            &mut self_ctx,
            &q_self,
            k_plane,
            v_plane,
            seq_len,
            self.n_text_head,
            head_dim,
            start_pos,
            end_pos,
            scale,
            &mut scores_scratch,
        )?;

        let mut proj_buf = vec![0.0_f32; total];
        project_rows(&mut proj_buf, &self_ctx, seq_len, d, &o_w, Some(&o_b))?;
        axpy_f32(hidden, &proj_buf, 1.0);
        Ok(())
    }

    fn project_and_cache_decoder_self_qkv(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        norm: &[f32],
        kv_view: &mut KvCacheView<'_>,
    ) -> Result<Vec<f32>, K3Error> {
        let d = self.n_text_state;
        let q_w = load_dec_2d(
            weights,
            layer,
            "attn.query.weight",
            "self_attn.q_proj.weight",
            d,
            d,
        )?;
        let q_b = load_dec_1d(
            weights,
            layer,
            "attn.query.bias",
            "self_attn.q_proj.bias",
            d,
        )?;
        let k_w = load_dec_2d(
            weights,
            layer,
            "attn.key.weight",
            "self_attn.k_proj.weight",
            d,
            d,
        )?;
        let k_b = load_dec_opt_1d(weights, layer, "attn.key.bias", "self_attn.k_proj.bias", d)?;
        let v_w = load_dec_2d(
            weights,
            layer,
            "attn.value.weight",
            "self_attn.v_proj.weight",
            d,
            d,
        )?;
        let v_b = load_dec_1d(
            weights,
            layer,
            "attn.value.bias",
            "self_attn.v_proj.bias",
            d,
        )?;

        let mut q_self = vec![0.0_f32; seq_len * d];
        let mut k_row = vec![0.0_f32; d];
        let mut v_row = vec![0.0_f32; d];
        for m in 0..seq_len {
            let x_row = &norm[m * d..(m + 1) * d];
            let q_dst = &mut q_self[m * d..(m + 1) * d];
            matvec_any(q_dst, &q_w, x_row)?;
            axpy_f32(q_dst, &q_b, 1.0);

            matvec_any(&mut k_row, &k_w, x_row)?;
            if let Some(kb) = k_b.as_deref() {
                axpy_f32(&mut k_row, kb, 1.0);
            }
            matvec_any(&mut v_row, &v_w, x_row)?;
            axpy_f32(&mut v_row, &v_b, 1.0);

            let (k_slot, v_slot) = kv_view.write_pos_range(layer, start_pos + m)?;
            k_slot.copy_from_slice(&k_row);
            v_slot.copy_from_slice(&v_row);
        }
        Ok(q_self)
    }

    fn run_decoder_cross_attn(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        seq_len: usize,
        enc_len: usize,
        cross_kv: &[f32],
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.n_text_state;
        let head_dim = self.text_head_dim();
        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let total = seq_len * d;

        let cross_ln_w = load_dec_1d(
            weights,
            layer,
            "cross_attn_ln.weight",
            "encoder_attn_layer_norm.weight",
            d,
        )?;
        let cross_ln_b = load_dec_1d(
            weights,
            layer,
            "cross_attn_ln.bias",
            "encoder_attn_layer_norm.bias",
            d,
        )?;
        let mut norm = hidden.to_vec();
        apply_layernorm_rows(
            &mut norm,
            seq_len,
            d,
            &cross_ln_w,
            &cross_ln_b,
            self.norm_eps,
        )?;

        let cross_q_w = load_dec_2d(
            weights,
            layer,
            "cross_attn.query.weight",
            "encoder_attn.q_proj.weight",
            d,
            d,
        )?;
        let cross_q_b = load_dec_1d(
            weights,
            layer,
            "cross_attn.query.bias",
            "encoder_attn.q_proj.bias",
            d,
        )?;
        let cross_out_w = load_dec_2d(
            weights,
            layer,
            "cross_attn.out.weight",
            "encoder_attn.out_proj.weight",
            d,
            d,
        )?;
        let cross_out_b = load_dec_1d(
            weights,
            layer,
            "cross_attn.out.bias",
            "encoder_attn.out_proj.bias",
            d,
        )?;

        let mut q_cross = vec![0.0_f32; total];
        project_rows(
            &mut q_cross,
            &norm,
            seq_len,
            d,
            &cross_q_w,
            Some(&cross_q_b),
        )?;

        let plane_len = enc_len * d;
        let layer_base = layer * 2 * plane_len;
        let k_enc = &cross_kv[layer_base..layer_base + plane_len];
        let v_enc = &cross_kv[layer_base + plane_len..layer_base + 2 * plane_len];

        let mut cross_ctx = vec![0.0_f32; total];
        cross_attention(
            &mut cross_ctx,
            &q_cross,
            k_enc,
            v_enc,
            seq_len,
            enc_len,
            self.n_text_head,
            head_dim,
            scale,
        )?;

        let mut proj_buf = vec![0.0_f32; total];
        project_rows(
            &mut proj_buf,
            &cross_ctx,
            seq_len,
            d,
            &cross_out_w,
            Some(&cross_out_b),
        )?;
        axpy_f32(hidden, &proj_buf, 1.0);
        Ok(())
    }

    fn run_decoder_mlp(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        seq_len: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.n_text_state;
        let ffn_dim = self.text_ffn_dim;

        let mlp_ln_w = load_dec_1d(
            weights,
            layer,
            "mlp_ln.weight",
            "final_layer_norm.weight",
            d,
        )?;
        let mlp_ln_b = load_dec_1d(weights, layer, "mlp_ln.bias", "final_layer_norm.bias", d)?;
        let mut norm = hidden.to_vec();
        apply_layernorm_rows(&mut norm, seq_len, d, &mlp_ln_w, &mlp_ln_b, self.norm_eps)?;

        let fc1_w = load_dec_mlp_2d(weights, layer, "0.weight", "fc1.weight", ffn_dim, d)?;
        let fc1_b = load_dec_mlp_1d(weights, layer, "0.bias", "fc1.bias", ffn_dim)?;
        let fc2_w = load_dec_mlp_2d(weights, layer, "2.weight", "fc2.weight", d, ffn_dim)?;
        let fc2_b = load_dec_mlp_1d(weights, layer, "2.bias", "fc2.bias", d)?;

        let mut ffn = vec![0.0_f32; seq_len * ffn_dim];
        for m in 0..seq_len {
            let x_row = &norm[m * d..(m + 1) * d];
            let ffn_row = &mut ffn[m * ffn_dim..(m + 1) * ffn_dim];
            matvec_any(ffn_row, &fc1_w, x_row)?;
            axpy_f32(ffn_row, &fc1_b, 1.0);
        }
        gelu_inplace(&mut ffn)?;

        let mut ffn_out = vec![0.0_f32; d];
        for m in 0..seq_len {
            let ffn_row = &ffn[m * ffn_dim..(m + 1) * ffn_dim];
            matvec_any(&mut ffn_out, &fc2_w, ffn_row)?;
            axpy_f32(&mut ffn_out, &fc2_b, 1.0);
            axpy_f32(&mut hidden[m * d..(m + 1) * d], &ffn_out, 1.0);
        }

        Ok(())
    }

    pub(crate) fn validate_weights_presence(
        &self,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_encoder_weights_presence(weights)?;
        self.validate_decoder_weights_presence(weights)
    }

    fn validate_encoder_weights_presence(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let d_enc = self.n_audio_state;
        let _ = resolve_tensor(
            weights,
            &["encoder.conv1.weight", "model.encoder.conv1.weight"],
        )?;
        let _ = resolve_tensor(weights, &["encoder.conv1.bias", "model.encoder.conv1.bias"])?;
        let _ = resolve_tensor(
            weights,
            &["encoder.conv2.weight", "model.encoder.conv2.weight"],
        )?;
        let _ = resolve_tensor(weights, &["encoder.conv2.bias", "model.encoder.conv2.bias"])?;
        let _ = require_embedding_table(
            weights,
            &[
                "encoder.positional_embedding",
                "encoder.positional_embedding.weight",
                "model.encoder.embed_positions.weight",
            ],
            1,
            d_enc,
        )?;
        let _ = resolve_tensor(
            weights,
            &["encoder.ln_post.weight", "model.encoder.layer_norm.weight"],
        )?;
        let _ = resolve_tensor(
            weights,
            &["encoder.ln_post.bias", "model.encoder.layer_norm.bias"],
        )?;

        for layer in 0..self.n_audio_layer {
            self.validate_encoder_layer_weights(weights, layer)?;
        }
        Ok(())
    }

    fn validate_encoder_layer_weights(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
    ) -> Result<(), K3Error> {
        let d_enc = self.n_audio_state;
        let _ = load_enc_1d(
            weights,
            layer,
            "attn_ln.weight",
            "self_attn_layer_norm.weight",
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "attn_ln.bias",
            "self_attn_layer_norm.bias",
            d_enc,
        )?;
        let _ = load_enc_2d(
            weights,
            layer,
            "attn.query.weight",
            "self_attn.q_proj.weight",
            d_enc,
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "attn.query.bias",
            "self_attn.q_proj.bias",
            d_enc,
        )?;
        let _ = load_enc_2d(
            weights,
            layer,
            "attn.key.weight",
            "self_attn.k_proj.weight",
            d_enc,
            d_enc,
        )?;
        let _ = load_enc_2d(
            weights,
            layer,
            "attn.value.weight",
            "self_attn.v_proj.weight",
            d_enc,
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "attn.value.bias",
            "self_attn.v_proj.bias",
            d_enc,
        )?;
        let _ = load_enc_2d(
            weights,
            layer,
            "attn.out.weight",
            "self_attn.out_proj.weight",
            d_enc,
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "attn.out.bias",
            "self_attn.out_proj.bias",
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "mlp_ln.weight",
            "final_layer_norm.weight",
            d_enc,
        )?;
        let _ = load_enc_1d(
            weights,
            layer,
            "mlp_ln.bias",
            "final_layer_norm.bias",
            d_enc,
        )?;
        let _ = load_enc_mlp_2d(
            weights,
            layer,
            "0.weight",
            "fc1.weight",
            self.audio_ffn_dim,
            d_enc,
        )?;
        let _ = load_enc_mlp_2d(
            weights,
            layer,
            "2.weight",
            "fc2.weight",
            d_enc,
            self.audio_ffn_dim,
        )?;
        Ok(())
    }

    fn validate_decoder_weights_presence(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let d_dec = self.n_text_state;
        let _ = require_embedding_table(
            weights,
            &[
                "decoder.token_embedding.weight",
                "decoder.token_embedding",
                "model.decoder.embed_tokens.weight",
                "token_embd.weight",
            ],
            self.n_vocab,
            d_dec,
        )?;
        let _ = require_embedding_table(
            weights,
            &[
                "decoder.positional_embedding",
                "decoder.positional_embedding.weight",
                "model.decoder.embed_positions.weight",
            ],
            1,
            d_dec,
        )?;
        let _ = resolve_tensor(
            weights,
            &[
                "decoder.ln.weight",
                "model.decoder.layer_norm.weight",
                "decoder.layer_norm.weight",
            ],
        )?;
        let _ = resolve_tensor(
            weights,
            &[
                "decoder.ln.bias",
                "model.decoder.layer_norm.bias",
                "decoder.layer_norm.bias",
            ],
        )?;

        for layer in 0..self.n_text_layer {
            self.validate_decoder_layer_attn_weights(weights, layer)?;
            self.validate_decoder_layer_cross_mlp_weights(weights, layer)?;
        }

        Ok(())
    }

    fn validate_decoder_layer_attn_weights(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
    ) -> Result<(), K3Error> {
        let d_dec = self.n_text_state;
        let _ = load_dec_1d(
            weights,
            layer,
            "attn_ln.weight",
            "self_attn_layer_norm.weight",
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "attn_ln.bias",
            "self_attn_layer_norm.bias",
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "attn.query.weight",
            "self_attn.q_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "attn.query.bias",
            "self_attn.q_proj.bias",
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "attn.key.weight",
            "self_attn.k_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "attn.value.weight",
            "self_attn.v_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "attn.value.bias",
            "self_attn.v_proj.bias",
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "attn.out.weight",
            "self_attn.out_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "attn.out.bias",
            "self_attn.out_proj.bias",
            d_dec,
        )?;
        Ok(())
    }

    fn validate_decoder_layer_cross_mlp_weights(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
    ) -> Result<(), K3Error> {
        let d_enc = self.n_audio_state;
        let d_dec = self.n_text_state;
        let _ = load_dec_1d(
            weights,
            layer,
            "cross_attn_ln.weight",
            "encoder_attn_layer_norm.weight",
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "cross_attn_ln.bias",
            "encoder_attn_layer_norm.bias",
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "cross_attn.query.weight",
            "encoder_attn.q_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "cross_attn.query.bias",
            "encoder_attn.q_proj.bias",
            d_dec,
        )?;
        let _ = load_cross_kv_2d(weights, layer, "key", "k_proj", d_dec, d_enc)?;
        let _ = load_cross_kv_2d(weights, layer, "value", "v_proj", d_dec, d_enc)?;
        let _ = load_dec_1d(
            weights,
            layer,
            "cross_attn.value.bias",
            "encoder_attn.v_proj.bias",
            d_dec,
        )?;
        let _ = load_dec_2d(
            weights,
            layer,
            "cross_attn.out.weight",
            "encoder_attn.out_proj.weight",
            d_dec,
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "cross_attn.out.bias",
            "encoder_attn.out_proj.bias",
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "mlp_ln.weight",
            "final_layer_norm.weight",
            d_dec,
        )?;
        let _ = load_dec_1d(
            weights,
            layer,
            "mlp_ln.bias",
            "final_layer_norm.bias",
            d_dec,
        )?;
        let _ = load_dec_mlp_2d(
            weights,
            layer,
            "0.weight",
            "fc1.weight",
            self.text_ffn_dim,
            d_dec,
        )?;
        let _ = load_dec_mlp_2d(
            weights,
            layer,
            "2.weight",
            "fc2.weight",
            d_dec,
            self.text_ffn_dim,
        )?;
        Ok(())
    }
}

impl ArchRecipe for WhisperRecipe {
    fn architecture_id(&self) -> &'static str {
        "whisper"
    }

    fn requires_kv_cache(&self) -> bool {
        true
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_config()?;
        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0 {
                let view = weights.tensor(desc.name())?;
                if !self.supports_quant(view.quant()) {
                    return Err(K3Error::unsupported_tensor_quant(
                        metadata.architecture(),
                        view.name(),
                        view.quant(),
                    ));
                }
            }
        }
        if weights.tensor_count() > 0 {
            self.validate_weights_presence(weights)?;
        }
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_config()?;

        if let Some(mel) = ctx.audio_mel() {
            let (n_mels, n_frames) =
                ctx.audio_dims()
                    .ok_or_else(|| K3Error::CapabilityMismatch {
                        architecture: "whisper".to_owned(),
                        reason: "audio_mel attached without valid audio_dims".to_owned(),
                    })?;
            let mel_owned = mel.to_vec();
            let cross_kv = ctx.cross_attn_kv_mut().ok_or_else(|| {
                K3Error::CapabilityMismatch {
                    architecture: "whisper".to_owned(),
                    reason: "audio_mel attached during prefill_step requires mutable cross_attn_kv via ExecutionContext::with_cross_attn_kv".to_owned(),
                }
            })?;
            self.encode_audio(weights, &mel_owned, n_mels, n_frames, cross_kv)?;
        }

        let prompt = ctx.prompt_tokens().to_vec();
        let start_pos = ctx.position();
        self.decode_tokens(ctx, weights, &prompt, start_pos)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_config()?;
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "whisper".to_owned(),
                reason: "decode_step called without a staged decode_token".to_owned(),
            })?;
        let pos = ctx.position();
        self.decode_tokens(ctx, weights, &[token], pos)
    }
}

fn apply_layernorm_rows(
    data: &mut [f32],
    rows: usize,
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let mut tmp = vec![0.0_f32; dim];
    for r in 0..rows {
        let row = &mut data[r * dim..(r + 1) * dim];
        layernorm(&mut tmp, row, Some(weight), Some(bias), eps)?;
        row.copy_from_slice(&tmp);
    }
    Ok(())
}

fn project_rows(
    out: &mut [f32],
    input: &[f32],
    rows: usize,
    dim: usize,
    weight: &TensorView<'_>,
    bias: Option<&[f32]>,
) -> Result<(), K3Error> {
    for r in 0..rows {
        let in_row = &input[r * dim..(r + 1) * dim];
        let out_row = &mut out[r * dim..(r + 1) * dim];
        matvec_any(out_row, weight, in_row)?;
        if let Some(b) = bias {
            axpy_f32(out_row, b, 1.0);
        }
    }
    Ok(())
}

fn load_enc_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Vec<f32>, K3Error> {
    let a = encoder_block_aliases(layer, suffix, hf_suffix);
    require_1d_f32(weights, &[a[0].as_str(), a[1].as_str()], len)
}

fn load_enc_opt_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    let a = encoder_block_aliases(layer, suffix, hf_suffix);
    require_optional_1d_f32(weights, &[a[0].as_str(), a[1].as_str()], len)
}

fn load_enc_2d<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let a = encoder_block_aliases(layer, suffix, hf_suffix);
    require_2d_matvec(weights, &[a[0].as_str(), a[1].as_str()], out_dim, in_dim)
}

fn load_dec_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Vec<f32>, K3Error> {
    let a = decoder_block_aliases(layer, suffix, hf_suffix);
    require_1d_f32(weights, &[a[0].as_str(), a[1].as_str()], len)
}

fn load_dec_opt_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    let a = decoder_block_aliases(layer, suffix, hf_suffix);
    require_optional_1d_f32(weights, &[a[0].as_str(), a[1].as_str()], len)
}

fn load_dec_2d<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let a = decoder_block_aliases(layer, suffix, hf_suffix);
    require_2d_matvec(weights, &[a[0].as_str(), a[1].as_str()], out_dim, in_dim)
}

fn load_enc_mlp_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Vec<f32>, K3Error> {
    let a = encoder_mlp_aliases(layer, suffix, hf_suffix);
    require_1d_f32(weights, &[a[0].as_str(), a[1].as_str(), a[2].as_str()], len)
}

fn load_enc_mlp_2d<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let a = encoder_mlp_aliases(layer, suffix, hf_suffix);
    require_2d_matvec(
        weights,
        &[a[0].as_str(), a[1].as_str(), a[2].as_str()],
        out_dim,
        in_dim,
    )
}

fn load_dec_mlp_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    len: usize,
) -> Result<Vec<f32>, K3Error> {
    let a = decoder_mlp_aliases(layer, suffix, hf_suffix);
    require_1d_f32(weights, &[a[0].as_str(), a[1].as_str(), a[2].as_str()], len)
}

fn load_dec_mlp_2d<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let a = decoder_mlp_aliases(layer, suffix, hf_suffix);
    require_2d_matvec(
        weights,
        &[a[0].as_str(), a[1].as_str(), a[2].as_str()],
        out_dim,
        in_dim,
    )
}

fn load_cross_kv_2d<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffix: &str,
    hf_suffix: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let a = cross_attn_kv_aliases(layer, suffix, hf_suffix);
    require_2d_matvec(
        weights,
        &[a[0].as_str(), a[1].as_str(), a[2].as_str()],
        out_dim,
        in_dim,
    )
}

fn gguf_usize(gguf: &GgufFile<'_>, keys: &[&str]) -> Option<usize> {
    for &key in keys {
        if let Some(val) = gguf
            .metadata_value(key)
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            return Some(val);
        }
    }
    None
}

fn infer_encoder_metadata(metadata: &ModelMetadata) -> (usize, usize, usize, usize, usize, usize) {
    let tensors = metadata.tensors();
    let (conv_out, conv_mels) = infer_conv1_dims(tensors);
    let enc_ln_dim = find_descriptor(
        tensors,
        &["encoder.ln_post.weight", "model.encoder.layer_norm.weight"],
    )
    .filter(|d| d.shape().len() == 1 && d.shape()[0] > 0)
    .map(|d| d.shape()[0]);

    let n_audio_state = conv_out
        .or(enc_ln_dim)
        .or_else(|| (metadata.embedding_dim() > 0).then_some(metadata.embedding_dim()))
        .unwrap_or(512);

    let mel_filter_bins =
        find_descriptor(tensors, &["filters.mel", "mel_filters"]).and_then(|d| match d.shape() {
            [m, b] | [b, m] if *b == 201 || *b == 257 => Some(*m),
            _ => None,
        });
    let n_mels = conv_mels
        .or(mel_filter_bins)
        .unwrap_or(if n_audio_state == 1280 { 128 } else { 80 });

    let n_audio_ctx = infer_table_rows(
        tensors,
        &[
            "encoder.positional_embedding",
            "encoder.positional_embedding.weight",
            "model.encoder.embed_positions.weight",
        ],
        n_audio_state,
    )
    .unwrap_or(1500);

    let n_audio_layer =
        infer_indexed_layer_count(tensors, &["encoder.blocks.", "model.encoder.layers."])
            .or_else(|| (metadata.layer_count() > 0).then_some(metadata.layer_count()))
            .unwrap_or(default_audio_layers_for_dim(n_audio_state));

    let meta_heads = metadata.head_count();
    let n_audio_head = if meta_heads > 0 && n_audio_state % meta_heads == 0 {
        meta_heads
    } else {
        infer_head_count(n_audio_state)
    };

    let audio_ffn_dim = infer_ffn_dim(
        tensors,
        &[
            "encoder.blocks.0.mlp.0.bias",
            "encoder.blocks.0.mlp.fc1.bias",
            "model.encoder.layers.0.fc1.bias",
        ],
        &[
            "encoder.blocks.0.mlp.0.weight",
            "encoder.blocks.0.mlp.fc1.weight",
            "model.encoder.layers.0.fc1.weight",
        ],
        n_audio_state,
    )
    .or_else(|| metadata.feed_forward_length())
    .unwrap_or(n_audio_state.saturating_mul(4));

    (
        n_mels,
        n_audio_ctx,
        n_audio_state,
        n_audio_head,
        n_audio_layer,
        audio_ffn_dim,
    )
}

fn infer_decoder_metadata(
    metadata: &ModelMetadata,
    n_mels: usize,
    n_audio_state: usize,
    n_audio_layer: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let tensors = metadata.tensors();
    let dec_ln_dim = find_descriptor(
        tensors,
        &[
            "decoder.ln.weight",
            "model.decoder.layer_norm.weight",
            "decoder.layer_norm.weight",
        ],
    )
    .filter(|d| d.shape().len() == 1 && d.shape()[0] > 0)
    .map(|d| d.shape()[0]);
    let n_text_state = dec_ln_dim.unwrap_or(n_audio_state);

    let tok_rows = infer_table_rows(
        tensors,
        &[
            "decoder.token_embedding.weight",
            "decoder.token_embedding",
            "model.decoder.embed_tokens.weight",
            "token_embd.weight",
        ],
        n_text_state,
    );
    let n_vocab = tok_rows
        .or_else(|| metadata.vocab_size())
        .unwrap_or(if n_mels == 128 {
            WHISPER_VOCAB_V3
        } else {
            WHISPER_VOCAB_V2
        });

    let n_text_ctx = infer_table_rows(
        tensors,
        &[
            "decoder.positional_embedding",
            "decoder.positional_embedding.weight",
            "model.decoder.embed_positions.weight",
        ],
        n_text_state,
    )
    .or_else(|| {
        let ctx = metadata.context_length();
        (ctx > 0 && ctx != 2048).then_some(ctx)
    })
    .unwrap_or(448);

    let n_text_layer =
        infer_indexed_layer_count(tensors, &["decoder.blocks.", "model.decoder.layers."])
            .or_else(|| (metadata.layer_count() > 0).then_some(metadata.layer_count()))
            .unwrap_or(if n_audio_state == 1280 && n_mels == 128 {
                4
            } else {
                n_audio_layer
            });

    let meta_heads = metadata.head_count();
    let n_text_head = if meta_heads > 0 && n_text_state % meta_heads == 0 {
        meta_heads
    } else {
        infer_head_count(n_text_state)
    };

    let text_ffn_dim = infer_ffn_dim(
        tensors,
        &[
            "decoder.blocks.0.mlp.0.bias",
            "decoder.blocks.0.mlp.fc1.bias",
            "model.decoder.layers.0.fc1.bias",
        ],
        &[
            "decoder.blocks.0.mlp.0.weight",
            "decoder.blocks.0.mlp.fc1.weight",
            "model.decoder.layers.0.fc1.weight",
        ],
        n_text_state,
    )
    .or_else(|| metadata.feed_forward_length())
    .unwrap_or(n_text_state.saturating_mul(4));

    (
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer,
        text_ffn_dim,
        n_vocab,
    )
}

fn validate_whisper_architecture(metadata: &ModelMetadata) -> Result<(), K3Error> {
    let arch = metadata.architecture().trim();
    let lower = arch.to_ascii_lowercase();
    if lower == "whisper" || lower.starts_with("whisper-") || lower.starts_with("whisper_") {
        return Ok(());
    }
    if lower == "unknown" {
        let has_whisper_tensor = find_descriptor(
            metadata.tensors(),
            &[
                "encoder.conv1.weight",
                "model.encoder.conv1.weight",
                "decoder.ln.weight",
                "model.decoder.layer_norm.weight",
            ],
        )
        .is_some();
        if has_whisper_tensor {
            return Ok(());
        }
    }
    Err(K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: format!("expected 'whisper' architecture metadata, got '{arch}'"),
    })
}

fn infer_conv1_dims(tensors: &[TensorDescriptor]) -> (Option<usize>, Option<usize>) {
    let Some(desc) = find_descriptor(
        tensors,
        &["encoder.conv1.weight", "model.encoder.conv1.weight"],
    ) else {
        return (None, None);
    };
    match desc.shape() {
        [out_ch, in_ch, 3] | [3, in_ch, out_ch] if *out_ch > 0 && *in_ch > 0 => {
            (Some(*out_ch), Some(*in_ch))
        }
        _ => (None, None),
    }
}

fn infer_table_rows(
    tensors: &[TensorDescriptor],
    aliases: &[&str],
    row_width: usize,
) -> Option<usize> {
    let desc = find_descriptor(tensors, aliases)?;
    let shape = desc.shape();
    if shape.len() != 2 || row_width == 0 {
        return None;
    }
    if shape[1] == row_width && shape[0] != row_width {
        Some(shape[0])
    } else if shape[0] == row_width {
        Some(shape[1])
    } else {
        None
    }
}

fn infer_ffn_dim(
    tensors: &[TensorDescriptor],
    bias_aliases: &[&str],
    weight_aliases: &[&str],
    d_model: usize,
) -> Option<usize> {
    if let Some(bias) = find_descriptor(tensors, bias_aliases)
        && bias.shape().len() == 1
        && bias.shape()[0] > 0
    {
        return Some(bias.shape()[0]);
    }
    if let Some(w) = find_descriptor(tensors, weight_aliases)
        && w.shape().len() == 2
    {
        let s = w.shape();
        if s[1] == d_model && s[0] > 0 {
            return Some(s[0]);
        }
        if s[0] == d_model && s[1] > 0 {
            return Some(s[1]);
        }
    }
    None
}

const fn default_audio_layers_for_dim(d_model: usize) -> usize {
    match d_model {
        384 => 4,
        768 => 12,
        1024 => 24,
        1280 => 32,
        _ => 6,
    }
}

const fn infer_head_count(d_model: usize) -> usize {
    if d_model >= 64 && d_model % 64 == 0 {
        d_model / 64
    } else if d_model % 8 == 0 {
        8
    } else if d_model % 4 == 0 {
        4
    } else if d_model % 2 == 0 {
        2
    } else {
        1
    }
}
