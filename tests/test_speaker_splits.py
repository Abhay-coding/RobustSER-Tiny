import unittest

from scripts.create_speaker_splits import create_assignments, validate


def make_rows():
    rows = []
    for speaker in range(1, 21):
        for emotion in ("Angry", "Happy", "Neutral"):
            rows.append({"filename": f"{speaker}_{emotion}.wav", "path": "unused", "speaker_id": str(speaker), "emotion": emotion, "sha256": f"{speaker}_{emotion}"})
    return rows


class SpeakerSplitTests(unittest.TestCase):
    def test_assignments_are_deterministic_and_speaker_disjoint(self):
        rows = make_rows()
        first = create_assignments(rows, seed=42)
        second = create_assignments(rows, seed=42)

        self.assertEqual(first, second)
        annotated = [{**row, "split": first[row["speaker_id"]]} for row in rows]
        summary = validate(annotated)
        self.assertEqual(20, sum(summary["speakers"].values()))
        self.assertEqual({"Angry", "Happy", "Neutral"}, set(summary["labels"]))

    def test_rejects_exact_audio_cross_split_overlap(self):
        rows = [
            {"filename": "a.wav", "speaker_id": "1", "emotion": "Angry", "sha256": "same", "split": "train"},
            {"filename": "b.wav", "speaker_id": "2", "emotion": "Angry", "sha256": "same", "split": "validation"},
            {"filename": "c.wav", "speaker_id": "3", "emotion": "Angry", "sha256": "third", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "Exact recording overlap"):
            validate(rows)

    def test_rejects_missing_paths_when_requested(self):
        rows = [
            {"filename": "a.wav", "path": "does-not-exist.wav", "speaker_id": "1", "emotion": "Angry", "sha256": "a", "split": "train"},
            {"filename": "b.wav", "path": "does-not-exist.wav", "speaker_id": "2", "emotion": "Angry", "sha256": "b", "split": "validation"},
            {"filename": "c.wav", "path": "does-not-exist.wav", "speaker_id": "3", "emotion": "Angry", "sha256": "c", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "paths do not resolve"):
            validate(rows, check_paths=True)
