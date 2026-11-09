import io
import unittest

import numpy as np
from scipy.io import wavfile

from robustser.features.log_mel import SAMPLE_RATE
from robustser.inference import MAX_DEMO_DURATION_SECONDS, decode_wav_bytes, resample_to_model_rate


class DemoInferenceTests(unittest.TestCase):
    def test_decodes_stereo_wav_and_downmixes_channels(self) -> None:
        samples = np.array([[32767, -32768], [16384, 16384]], dtype=np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, 48_000, samples)
        audio, sample_rate = decode_wav_bytes(buffer.getvalue())
        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(audio.shape, (2,))
        self.assertAlmostEqual(float(audio[0]), 0.0, places=3)

    def test_resamples_browser_rate_to_model_rate(self) -> None:
        audio = np.sin(np.linspace(0.0, 10.0, 48_000, endpoint=False)).astype(np.float32)
        converted = resample_to_model_rate(audio, 48_000)
        self.assertEqual(converted.shape, (SAMPLE_RATE,))
        self.assertTrue(np.isfinite(converted).all())

    def test_rejects_empty_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_wav_bytes(b"")

    def test_rejects_impractically_long_demo_recording(self) -> None:
        samples = np.zeros(int((MAX_DEMO_DURATION_SECONDS + 1) * 16_000), dtype=np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, 16_000, samples)
        with self.assertRaisesRegex(ValueError, "no longer"):
            decode_wav_bytes(buffer.getvalue())
