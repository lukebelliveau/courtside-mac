"""Regression tests for BoT-SORT's shared, unconfirmed -1 sentinel.

Run with: python -m unittest discover -s . -p 'test_*.py'
No detector weights, network, or GPU are needed. The saved-output audit also
checks the latest demo, when present, for invalid identities reaching metadata.
"""

import json
from pathlib import Path
import re
import unittest

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

from analyze import confirmed_tracks


def detections(boxes, confidence):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.full(len(boxes), 2, dtype=int),
    )


class TrackingIdentityTests(unittest.TestCase):
    def test_real_tracker_shared_sentinel_never_becomes_a_player_identity(self):
        tracker = BoTSORTTracker(
            frame_rate=30, track_activation_threshold=.5,
            high_conf_det_threshold=.4, minimum_consecutive_frames=2,
            enable_cmc=False,
        )
        # Two distant, weak detections share -1 in the real upstream API.
        # Their reversed positions on the next frame must create no trajectory.
        weak_frames = [
            [[10, 10, 50, 130], [700, 50, 750, 170]],
            [[698, 50, 748, 170], [12, 10, 52, 130]],
        ]
        for frame, boxes in enumerate(weak_frames):
            raw = tracker.update(detections(boxes, [.2, .3]), timestamp=frame / 30)
            np.testing.assert_array_equal(raw.tracker_id, [-1, -1])
            self.assertEqual(len(confirmed_tracks(raw)), 0)
            # The boundary filter must not corrupt the upstream detection object.
            np.testing.assert_array_equal(raw.tracker_id, [-1, -1])

        # Strong detections appear, survive confirmation, and retain two unique
        # IDs when another unrelated unconfirmed detection enters the frame.
        raw = tracker.update(detections([[14, 10, 54, 130], [696, 50, 746, 170]], [.9, .9]), timestamp=2 / 30)
        np.testing.assert_array_equal(raw.tracker_id, [-1, -1])
        self.assertEqual(len(confirmed_tracks(raw)), 0)
        raw = tracker.update(detections([[16, 10, 56, 130], [694, 50, 744, 170]], [.9, .9]), timestamp=3 / 30)
        stable = confirmed_tracks(raw)
        self.assertEqual(len(stable), 2)
        self.assertTrue((stable.tracker_id >= 0).all())
        self.assertEqual(len(set(stable.tracker_id)), 2)

        raw = tracker.update(detections(
            [[18, 10, 58, 130], [692, 50, 742, 170], [400, 10, 450, 130]], [.9, .9, .2],
        ), timestamp=4 / 30)
        self.assertEqual(len(raw), 3)
        self.assertEqual(int((raw.tracker_id == -1).sum()), 1)
        filtered = confirmed_tracks(raw)
        np.testing.assert_array_equal(filtered.tracker_id, stable.tracker_id)
        np.testing.assert_allclose(filtered.xyxy, [[18, 10, 58, 130], [692, 50, 742, 170]])

    def test_corrupt_duplicate_confirmed_ids_fail_before_metadata(self):
        broken = detections([[10, 10, 50, 130], [700, 50, 750, 170]], [.9, .9])
        broken.tracker_id = np.array([4, 4])
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            confirmed_tracks(broken)

    def test_saved_demo_contains_only_unique_confirmed_ids(self):
        artifact = Path(__file__).resolve().parent / "outputs" / "demo" / "frames.jsonl"
        if not artifact.exists():
            self.skipTest("No generated demo artifact yet")
        frames, observations = 0, 0
        with artifact.open() as handle:
            for line in handle:
                row = json.loads(line)
                ids = [player["track"] for player in row["players"]]
                context = f"clip={row['clip']} frame={row['frame']}"
                self.assertEqual(len(ids), len(set(ids)), f"Duplicate identities at {context}")
                for track in ids:
                    self.assertIsNone(re.search(r"-t-\d+", track), f"Unconfirmed sentinel leaked at {context}: {track}")
                if row.get("near_ball") is not None:
                    self.assertIn(row["near_ball"], ids, f"Ball proximity references missing identity at {context}")
                frames += 1
                observations += len(ids)
        self.assertGreater(frames, 0, "Generated artifact has no frames")
        self.assertGreater(observations, 0, "Generated artifact has no tracked players")


if __name__ == "__main__":
    unittest.main()
