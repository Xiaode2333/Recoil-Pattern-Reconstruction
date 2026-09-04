from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_reconstruct import discover_videos, safe_stem


class BatchReconstructTests(unittest.TestCase):
    def test_discovery_is_case_insensitive_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "zeta.mp4").touch()
            (root / "Alpha.mp4").touch()
            self.assertEqual(
                [path.name for path in discover_videos(root, "*.mp4")],
                ["Alpha.mp4", "zeta.mp4"],
            )

    def test_safe_stem_removes_path_unfriendly_punctuation(self) -> None:
        self.assertEqual(safe_stem(Path("Capture #1 (ADS).mp4")), "capture-1-ads")

    def test_missing_inputs_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No videos"):
                discover_videos(Path(directory), "*.mp4")


if __name__ == "__main__":
    unittest.main()
