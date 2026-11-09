import unittest

from scripts.audit_audio_quality import dbfs, decode_pcm, robust_outliers


class AudioQualityHelpersTests(unittest.TestCase):
    def test_decodes_little_endian_16_bit_pcm(self):
        values, full_scale = decode_pcm(bytes([0, 0, 255, 127, 0, 128]), 2)

        self.assertEqual([0, 32767, -32768], values.tolist())
        self.assertEqual(32768.0, full_scale)

    def test_dbfs_of_full_scale_is_zero(self):
        self.assertAlmostEqual(0.0, dbfs(1.0))

    def test_detects_robust_outlier(self):
        _, _, outliers = robust_outliers([0.9, 1.0, 1.1, 1.2, 9.0])

        self.assertEqual({4}, outliers)
