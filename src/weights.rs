//! Fail-closed Whisper weight resolution, shape validation, and metadata inspection helpers.

use jero_core::{K3Error, ModelWeights, TensorDescriptor, TensorView};

use crate::ops::{bounds_err, checked_mul, dequantize_all_finite, is_supported_quant};

pub(crate) fn resolve_tensor<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<TensorView<'w>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    let canonical = aliases.first().copied().unwrap_or("unknown");
    weights.tensor(canonical)
}

pub(crate) fn resolve_optional_tensor<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<Option<TensorView<'w>>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name).map(Some);
        }
    }
    Ok(None)
}

pub(crate) fn require_1d_f32(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = resolve_tensor(weights, aliases)?;
    if view.shape().len() != 1 || view.num_elements() != expected_len {
        return Err(bounds_err(
            view.name(),
            expected_len,
            view.num_elements(),
            format!(
                "1D tensor '{}' expected shape [{expected_len}], got {:?}",
                view.name(),
                view.shape()
            ),
        ));
    }
    dequantize_all_finite(&view)
}

pub(crate) fn require_optional_1d_f32(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    let Some(view) = resolve_optional_tensor(weights, aliases)? else {
        return Ok(None);
    };
    if view.shape().len() != 1 || view.num_elements() != expected_len {
        return Err(bounds_err(
            view.name(),
            expected_len,
            view.num_elements(),
            format!(
                "1D tensor '{}' expected shape [{expected_len}], got {:?}",
                view.name(),
                view.shape()
            ),
        ));
    }
    dequantize_all_finite(&view).map(Some)
}

pub(crate) fn require_2d_matvec<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
    out_dim: usize,
    in_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let view = resolve_tensor(weights, aliases)?;
    if !is_supported_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "whisper",
            view.name(),
            view.quant(),
        ));
    }
    let expected_elems = checked_mul(view.name(), out_dim, in_dim)?;
    let shape = view.shape();
    let valid_2d = shape.len() == 2
        && ((shape[0] == out_dim && shape[1] == in_dim)
            || (shape[0] == in_dim && shape[1] == out_dim));
    if !valid_2d || view.num_elements() != expected_elems {
        return Err(bounds_err(
            view.name(),
            expected_elems,
            view.num_elements(),
            format!(
                "2D weight '{}' expected [{out_dim}, {in_dim}] or [{in_dim}, {out_dim}], got {shape:?}",
                view.name()
            ),
        ));
    }
    Ok(view)
}

pub(crate) fn require_3d_conv1d_f32(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    out_channels: usize,
    in_channels: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = resolve_tensor(weights, aliases)?;
    let oc_ic = checked_mul(view.name(), out_channels, in_channels)?;
    let expected_elems = checked_mul(view.name(), oc_ic, 3)?;
    let shape = view.shape();
    let valid_3d = shape.len() == 3
        && ((shape[0] == out_channels && shape[1] == in_channels && shape[2] == 3)
            || (shape[0] == 3 && shape[1] == in_channels && shape[2] == out_channels));
    if !valid_3d || view.num_elements() != expected_elems {
        return Err(bounds_err(
            view.name(),
            expected_elems,
            view.num_elements(),
            format!(
                "Conv1d weight '{}' expected [{out_channels}, {in_channels}, 3] or [3, {in_channels}, {out_channels}], got {shape:?}",
                view.name()
            ),
        ));
    }
    dequantize_all_finite(&view)
}

pub(crate) fn require_embedding_table<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
    min_rows: usize,
    row_width: usize,
) -> Result<(TensorView<'w>, usize), K3Error> {
    let view = resolve_tensor(weights, aliases)?;
    if !is_supported_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "whisper",
            view.name(),
            view.quant(),
        ));
    }
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(bounds_err(
            view.name(),
            checked_mul(view.name(), min_rows, row_width)?,
            view.num_elements(),
            format!(
                "embedding table '{}' expected 2D shape, got {shape:?}",
                view.name()
            ),
        ));
    }
    let num_rows = if shape[1] == row_width {
        shape[0]
    } else if shape[0] == row_width {
        shape[1]
    } else {
        return Err(bounds_err(
            view.name(),
            checked_mul(view.name(), min_rows, row_width)?,
            view.num_elements(),
            format!(
                "embedding table '{}' shape {shape:?} does not match row_width={row_width}",
                view.name()
            ),
        ));
    };
    if num_rows < min_rows {
        return Err(bounds_err(
            view.name(),
            checked_mul(view.name(), min_rows, row_width)?,
            view.num_elements(),
            format!(
                "embedding table '{}' has {num_rows} rows, fewer than required {min_rows}",
                view.name()
            ),
        ));
    }
    Ok((view, num_rows))
}

pub(crate) fn find_descriptor<'a>(
    descriptors: &'a [TensorDescriptor],
    aliases: &[&str],
) -> Option<&'a TensorDescriptor> {
    for &name in aliases {
        if let Some(desc) = descriptors.iter().find(|t| t.name() == name) {
            return Some(desc);
        }
    }
    None
}

pub(crate) fn infer_indexed_layer_count(
    descriptors: &[TensorDescriptor],
    prefixes: &[&str],
) -> Option<usize> {
    let mut max_idx: Option<usize> = None;
    for desc in descriptors {
        for &prefix in prefixes {
            if let Some(rest) = desc.name().strip_prefix(prefix)
                && let Some((idx_str, _)) = rest.split_once('.')
                && let Ok(idx) = idx_str.parse::<usize>()
            {
                max_idx = Some(max_idx.map_or(idx, |cur| cur.max(idx)));
            }
        }
    }
    max_idx.map(|idx| idx.saturating_add(1))
}

pub(crate) fn encoder_block_aliases(layer: usize, suffix: &str, hf_suffix: &str) -> [String; 2] {
    [
        format!("encoder.blocks.{layer}.{suffix}"),
        format!("model.encoder.layers.{layer}.{hf_suffix}"),
    ]
}

pub(crate) fn encoder_mlp_aliases(layer: usize, idx: &str, hf_name: &str) -> [String; 3] {
    [
        format!("encoder.blocks.{layer}.mlp.{idx}"),
        format!("encoder.blocks.{layer}.mlp.{hf_name}"),
        format!("model.encoder.layers.{layer}.{hf_name}"),
    ]
}

pub(crate) fn decoder_block_aliases(layer: usize, suffix: &str, hf_suffix: &str) -> [String; 2] {
    [
        format!("decoder.blocks.{layer}.{suffix}"),
        format!("model.decoder.layers.{layer}.{hf_suffix}"),
    ]
}

pub(crate) fn decoder_mlp_aliases(layer: usize, idx: &str, hf_name: &str) -> [String; 3] {
    [
        format!("decoder.blocks.{layer}.mlp.{idx}"),
        format!("decoder.blocks.{layer}.mlp.{hf_name}"),
        format!("model.decoder.layers.{layer}.{hf_name}"),
    ]
}

pub(crate) fn cross_attn_kv_aliases(layer: usize, kind: &str, hf_name: &str) -> [String; 3] {
    [
        format!("decoder.blocks.{layer}.cross_attn.{kind}.weight"),
        format!("decoder.blocks.{layer}.cross_attn.{kind}"),
        format!("model.decoder.layers.{layer}.encoder_attn.{hf_name}.weight"),
    ]
}
