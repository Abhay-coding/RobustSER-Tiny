import unittest

from scripts.create_modeling_manifest import build_manifests


def row(filename, speaker, emotion, digest, valid="True"):
    return {
        "filename": filename,
        "speaker_id": speaker,
        "emotion": emotion,
        "sha256": digest,
        "valid": valid,
    }


class CreateModelingManifestTests(unittest.TestCase):
    def test_excludes_all_members_of_label_conflicting_duplicate_group(self):
        rows = [
            row("a.wav", "1001", "Angry", "conflict"),
            row("b.wav", "1001", "Fear", "conflict"),
            row("c.wav", "1002", "Happy", "unique"),
        ]

        conflicts, eligible = build_manifests(rows)

        self.assertEqual(2, len(conflicts))
        self.assertEqual({"exact_audio_label_conflict"}, {item["reason"] for item in conflicts})
        self.assertEqual(["c.wav"], [item["filename"] for item in eligible])

    def test_retains_same_label_duplicate_but_reports_it_for_future_grouping(self):
        rows = [
            row("a.wav", "1001", "Angry", "duplicate"),
            row("b.wav", "1001", "Angry", "duplicate"),
        ]

        conflicts, eligible = build_manifests(rows)

        self.assertEqual(2, len(eligible))
        self.assertEqual(
            {"retain_as_group_for_split_leakage_prevention"},
            {item["disposition"] for item in conflicts},
        )

    def test_excludes_structurally_invalid_rows(self):
        conflicts, eligible = build_manifests([row("bad.wav", "1001", "Angry", "unique", valid="False")])

        self.assertEqual([], conflicts)
        self.assertEqual([], eligible)
