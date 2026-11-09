import unittest

import numpy as np

from src.robustser.features.log_mel import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    extract_log_mel,
    extract_mfcc,
    fit_audio_window,
    mel_filterbank,
)


class LogMelTests(unittest.TestCase):
    def test_center_crop_and_padding_have_fixed_length(self):
        short = np.array([1.0, 2.0], dtype=np.float32)
        long = np.arange(WINDOW_SAMPLES + 4, dtype=np.float32)

        padded = fit_audio_window(short)
        cropped = fit_audio_window(long)

        self.assertEqual(WINDOW_SAMPLES, padded.size)
        self.assertTrue(np.array_equal(np.array([1.0, 2.0], dtype=np.float32), padded[:2]))
        self.assertTrue(np.array_equal(np.array([2.0, 3.0], dtype=np.float32), cropped[:2]))

    def test_seeded_random_crop_is_reproducible(self):
        audio = np.arange(WINDOW_SAMPLES + 100, dtype=np.float32)

        first = fit_audio_window(audio, policy="seeded_random_crop_or_zero_pad", rng=np.random.default_rng(7))
        second = fit_audio_window(audio, policy="seeded_random_crop_or_zero_pad", rng=np.random.default_rng(7))

        self.assertTrue(np.array_equal(first, second))

    def test_log_mel_meets_e0_shape_and_finiteness_contract(self):
        time = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        sine = 0.5 * np.sin(2 * np.pi * 440 * time)

        features = extract_log_mel(sine)

        self.assertEqual((32, 31), features.shape)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(0.0, float(features.mean()), places=5)
        self.assertAlmostEqual(1.0, float(features.std()), places=5)

    def test_filterbank_shape(self):
        self.assertEqual((32, 257), mel_filterbank().shape)

    def test_mfcc_preserves_e3_input_contract(self):
        time = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        features = extract_mfcc(0.5 * np.sin(2 * np.pi * 440 * time))
        self.assertEqual((32, 31), features.shape)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(0.0, float(features.mean()), places=5)
        self.assertAlmostEqual(1.0, float(features.std()), places=5)

    def test_two_second_mfcc_has_62_frames(self):
        features = extract_mfcc(np.zeros(WINDOW_SAMPLES * 2, dtype=np.float32))
        self.assertEqual((32, 62), features.shape)
