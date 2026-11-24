//! `jero-arch-whisper`: Encoder-decoder speech recognition recipe (Whisper ASR) with cross-attention KV cache for the `SpacemiT` K3 platform.

#![deny(unsafe_op_in_unsafe_fn)]

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

    #[cfg(feature = "with-kernels")]
    #[test]
    fn bootstrap_links_kernels_crate() {
        let mut buf = [3.0_f32, 4.0];
        jero_kernels_rvv::l2_norm_f32_in_place(&mut buf, 1e-6);
        assert!((buf[0] - 0.6).abs() < 1e-5);
        assert!((buf[1] - 0.8).abs() < 1e-5);
    }
}
