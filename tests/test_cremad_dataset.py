import unittest
from pathlib import Path

import torch

from robustser.data.cremad import CREMADSERDataset, NoisyCREMADSERTestDataset, TASK_LABELS, load_split_rows
from robustser.data.ravdess import read_resampled_pcm16_mono


MANIFEST = Path("data/manifests/crema_d_speaker_splits.csv")


class CREMADatasetTests(unittest.TestCase):
    def test_task_label_spaces_are_explicit(self):
        self.assertEqual(5, len(TASK_LABELS["five_emotion_common"]))
        self.assertEqual(6, len(TASK_LABELS["crema_d_six_emotion"]))
        self.assertNotIn("Sad", TASK_LABELS["five_emotion_common"])

    def test_five_emotion_task_excludes_sad_without_relabeling(self):
        rows = load_split_rows(MANIFEST, split="test", task="five_emotion_common")
        self.assertNotIn("Sad", {row["emotion"] for row in rows})
        self.assertEqual(884, len(rows))

    def test_dataset_returns_baseline_feature_and_valid_label(self):
        dataset = CREMADSERDataset(MANIFEST, split="validation", task="crema_d_six_emotion")
        features, label = dataset[0]
        self.assertEqual((1, 32, 31), tuple(features.shape))
        self.assertEqual(torch.float32, features.dtype)
        self.assertIn(label, range(6))

    def test_noisy_test_dataset_uses_evaluation_holdout_noise(self):
        dataset = NoisyCREMADSERTestDataset(
            MANIFEST, task="crema_d_six_emotion", noise_manifest_path="data/manifests/musan_noise_splits.csv",
            snr_db=10.0, seed=7,
        )
        self.assertEqual(1066, len(dataset))
        self.assertTrue(all(row["noise_split"] == "evaluation_holdout" for row in dataset.noise_rows))

    def test_ravdess_resampling_returns_finite_16khz_audio(self):
        audio = read_resampled_pcm16_mono("dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav")
        self.assertGreater(audio.size, 16000)
        self.assertTrue(torch.isfinite(torch.from_numpy(audio)).all())
