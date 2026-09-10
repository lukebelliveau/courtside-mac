"""Meaningful geometric rejection checks; no model/network access required."""

import unittest

import numpy as np

from court import (
    COURT_LENGTH_CM, COURT_WIDTH_CM, CourtMapper, MODEL_KEYPOINTS_CM,
    _transform, fit_homography, render_court,
)


class CourtGeometryTests(unittest.TestCase):
    def setUp(self):
        self.court = np.array([
            [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
            [0.1, 0.8], [0.5, 0.8], [0.9, 0.8],
            [0.2, 0.4], [0.7, 0.5],
        ])
        self.camera = np.array([[1000, 190, 80], [-30, 480, 70], [0.1, 0.2, 1.]])
        self.pixels = _transform(self.court, self.camera)

    def test_recovers_perspective_with_a_bad_landmark(self):
        corrupted = self.pixels.copy()
        corrupted[-1] += [180, 80]
        fit = fit_homography(corrupted, self.court, (720, 1280, 3), 10.0)
        self.assertTrue(fit.valid, fit.reason)
        self.assertEqual(fit.inliers, 7)
        np.testing.assert_allclose(_transform(self.pixels, fit.image_to_court), self.court, atol=1e-5)

    def test_insufficient_landmarks_rejected(self):
        fit = fit_homography(self.pixels[:4], self.court[:4], (720, 1280), 0.)
        self.assertFalse(fit.valid)

    def test_collinear_or_duplicate_landmarks_rejected(self):
        line = np.column_stack((np.linspace(0.1, .9, 8), np.full(8, .5)))
        fit = fit_homography(_transform(line, self.camera), line, (720, 1280), 0.)
        self.assertFalse(fit.valid)
        duplicate = np.repeat(self.pixels[:1], 8, axis=0)
        self.assertFalse(fit_homography(duplicate, self.court, (720, 1280), 0.).valid)

    def test_expired_future_and_off_court_positions_are_missing(self):
        mapper = CourtMapper(max_age_seconds=.2)
        mapper.fit = fit_homography(self.pixels, self.court, (720, 1280), 10.)
        np.testing.assert_allclose(mapper.project(self.pixels[:2], 10.1), self.court[:2], atol=1e-5)
        self.assertTrue(np.isnan(mapper.project(self.pixels[:2], 10.3)).all())
        self.assertTrue(np.isnan(mapper.project(self.pixels[:2], 9.9)).all())
        off_court = _transform(np.array([[-.1, .5], [1.1, .5], [.5, -.1], [.5, 1.1]]), self.camera)
        self.assertTrue(np.isnan(mapper.project(off_court, 10.1)).all())

    def test_failed_update_and_cut_invalidate_previous_fit(self):
        mapper = CourtMapper()
        mapper.fit = fit_homography(self.pixels, self.court, (720, 1280), 10.)
        mapper.update_keypoints(np.zeros((48, 2)), np.zeros(48), (720, 1280), 10.05)
        self.assertFalse(mapper.fit.valid)
        self.assertTrue(np.isnan(mapper.project(self.pixels, 10.05)).all())
        mapper.fit = fit_homography(self.pixels, self.court, (720, 1280), 11.)
        mapper.reset()
        self.assertTrue(np.isnan(mapper.project(self.pixels, 11.)).all())

    def test_verified_schema_excludes_unknown_and_nonpainted_slots(self):
        self.assertTrue({8, 24, 28, 34, 41, 42}.isdisjoint(MODEL_KEYPOINTS_CM))
        self.assertEqual(MODEL_KEYPOINTS_CM[0], (0, 0))
        self.assertEqual(MODEL_KEYPOINTS_CM[39], (COURT_LENGTH_CM, 1006))
        self.assertEqual(MODEL_KEYPOINTS_CM[18], (835, COURT_WIDTH_CM))

    def test_render_has_court_and_rejects_nonfinite_markers(self):
        clean = render_court(width=330)
        invalid = render_court(np.array([[np.nan, .2], [-.1, .5]]), width=330)
        self.assertEqual(clean.shape[1], 330)
        self.assertGreater(np.unique(clean.reshape(-1, 3), axis=0).shape[0], 20)
        np.testing.assert_array_equal(clean, invalid)


if __name__ == "__main__":
    unittest.main()
