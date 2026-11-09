import unittest

import numpy as np

from robustser.augmentation.noise import mix_at_snr, noise_segment


class NoiseAugmentationTests(unittest.TestCase):
    def test_noise_segment_repeats_short_source(self):
        segment = noise_segment(np.array([1.0, 2.0], dtype=np.float32), length=5, rng=np.random.default_rng(1))
        self.assertEqual([1.0, 2.0, 1.0, 2.0, 1.0], segment.tolist())

    def test_mixer_achieves_requested_snr_without_clipping(self):
        clean = np.full(16000, 0.1, dtype=np.float32)
        noise = np.tile(np.array([-0.1, 0.1], dtype=np.float32), 8000)
        mixed = mix_at_snr(clean, noise, snr_db=10.0, rng=np.random.default_rng(1))
        added_noise = mixed - clean
        observed_snr = 20 * np.log10(np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(added_noise**2)))
        self.assertAlmostEqual(10.0, float(observed_snr), places=4)
