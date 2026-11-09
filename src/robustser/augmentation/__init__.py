"""Controlled, reproducible acoustic augmentations."""

from .noise import mix_at_snr, noise_rows_from_manifest

__all__ = ["mix_at_snr", "noise_rows_from_manifest"]
