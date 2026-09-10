"""Court keypoints, conservative perspective calibration, and a real NBA court.

Coordinates use x along the court's 94 ft length and y along its 50 ft width.
The public projection API returns normalized (x, y) in [0, 1]. Missing or
unreliable positions are NaN, never snapped to a court boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


# NBA geometry from Roboflow sports/basketball/config.py, in centimeters.
# https://github.com/roboflow/sports/blob/feat/basketball/sports/basketball/config.py
COURT_LENGTH_CM = 2865.0
COURT_WIDTH_CM = 1524.0
PAINT_LENGTH_CM = 579.0
PAINT_WIDTH_CM = 488.0
RIM_X_CM = 160.0
ARC_RADIUS_CM = 724.0
CORNER_Y_CM = 91.0
FREE_THROW_RADIUS_CM = 183.0

# This model's 48-slot export is NOT the standard 33-vertex Roboflow ordering.
# Pairings below were checked against eight original image/annotation pairs in:
# https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8
# Exclude unused slots, symmetry-only inferences 28/41/42, and the unpainted
# basket floor projections 8/34. No elevated hoop points enter the plane fit.
# The retained slot-to-floor mapping is recorded explicitly below.
MODEL_KEYPOINTS_CM: dict[int, tuple[float, float]] = {
    0: (0, 0), 1: (0, 91), 3: (0, 518), 4: (0, 1006),
    6: (0, 1433), 7: (0, 1524), 9: (424, 91), 11: (424, 1433),
    12: (579, 518), 13: (579, 762), 14: (579, 1006),
    16: (835, 0), 17: (884, 762), 18: (835, 1524),
    20: (1432, 0), 22: (1432, 762),
    26: (2030, 0), 27: (1981, 762),
    29: (2286, 518), 30: (2286, 762), 31: (2286, 1006),
    32: (2441, 91), 33: (2441, 1433),
    35: (2865, 0), 36: (2865, 91), 38: (2865, 518), 39: (2865, 1006),
}
MODEL_MAPPING_SOURCE = (
    "https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8; "
    "48-slot annotations visually checked against Roboflow NBA court geometry"
)


@dataclass
class CourtFit:
    valid: bool
    reason: str
    timestamp: float
    keypoints: int = 0
    inliers: int = 0
    reprojection_rmse_px: float | None = None
    image_to_court: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "keypoints": self.keypoints,
            "inliers": self.inliers,
            "reprojection_rmse_px": self.reprojection_rmse_px,
        }


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    good = np.isfinite(hom).all(axis=1) & (np.abs(hom[:, 2]) > 1e-9)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[good] = hom[good, :2] / hom[good, 2:3]
    return result


def _spread(points: np.ndarray) -> float:
    """Convex-hull area, zero for repeated or collinear points."""
    if len(points) < 3 or not np.isfinite(points).all():
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.float32(points))))


def fit_homography(
    points_px: np.ndarray,
    points_normalized: np.ndarray,
    frame_shape: Sequence[int],
    timestamp: float,
    *,
    min_points: int = 6,
    min_inlier_ratio: float = 0.65,
) -> CourtFit:
    """Robustly fit image pixels to a court, with independent geometric checks.

    Six landmarks are required by default, although four mathematically suffice:
    four arbitrary points can always fit perfectly and provide no redundancy.
    RANSAC runs court-to-image so its error threshold has a pixel interpretation.
    """
    source = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(points_normalized, dtype=np.float64).reshape(-1, 2)
    count = len(source)

    def rejected(reason: str) -> CourtFit:
        return CourtFit(False, reason, timestamp, keypoints=count)

    if len(target) != count:
        return rejected("landmark count mismatch")
    if not np.isfinite(timestamp):
        return rejected("invalid timestamp")
    if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        return rejected("invalid frame dimensions")
    height, width = frame_shape[:2]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    inside = (
        (source[:, 0] >= 0) & (source[:, 0] < width)
        & (source[:, 1] >= 0) & (source[:, 1] < height)
        & (target >= 0).all(axis=1) & (target <= 1).all(axis=1)
    )
    source, target = source[finite & inside], target[finite & inside]
    count = len(source)
    if count < max(4, min_points):
        return rejected("insufficient visible landmarks")
    if len(np.unique(np.round(source, 2), axis=0)) < min_points:
        return rejected("duplicate image landmarks")
    if len(np.unique(target, axis=0)) < min_points:
        return rejected("duplicate court landmarks")
    if _spread(source / [width, height]) < 0.006 or _spread(target) < 0.02:
        return rejected("landmarks are collinear or cover too little area")

    tolerance_px = max(4.0, float(np.hypot(width, height)) * 0.006)
    try:
        court_to_image, mask = cv2.findHomography(
            target, source, cv2.RANSAC, tolerance_px,
            maxIters=3000, confidence=0.995,
        )
    except cv2.error:
        return rejected("homography estimation failed")
    if court_to_image is None or mask is None or not np.isfinite(court_to_image).all():
        return rejected("homography estimation failed")
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < min_points or inlier_count / count < min_inlier_ratio:
        return rejected("too few consistent landmarks")
    if _spread(source[inlier_mask] / [width, height]) < 0.006 or _spread(target[inlier_mask]) < 0.02:
        return rejected("inlier geometry is degenerate")

    # Remove pixel units before testing numerical conditioning.
    scaled = np.diag([1 / width, 1 / height, 1.0]) @ court_to_image
    if not np.isfinite(np.linalg.cond(scaled)) or np.linalg.cond(scaled) > 1e5:
        return rejected("unstable perspective transform")
    try:
        image_to_court = np.linalg.inv(court_to_image)
    except np.linalg.LinAlgError:
        return rejected("singular perspective transform")
    estimated = _transform(target[inlier_mask], court_to_image)
    errors = np.linalg.norm(estimated - source[inlier_mask], axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    if not np.isfinite(rmse) or rmse > tolerance_px or float(errors.max()) > 2 * tolerance_px:
        return rejected("landmark reprojection error too large")

    # A horizon through the landmark support invalidates its interior geometry.
    hull = cv2.convexHull(np.float32(source[inlier_mask])).reshape(-1, 2)
    denominators = np.column_stack((hull, np.ones(len(hull)))) @ image_to_court[2]
    if not ((denominators > 1e-9).all() or (denominators < -1e-9).all()):
        return rejected("perspective horizon crosses court support")
    return CourtFit(True, "calibrated", timestamp, count, inlier_count, rmse, image_to_court)


class CourtMapper:
    """Predict court landmarks on MPS and maintain a short-lived calibration.

    Every failed update invalidates the old fit immediately. Successful fits
    expire after max_age_seconds; reset at cuts to prevent cross-scene reuse.
    Only floor-plane landmarks belong in keypoints_cm (never an elevated rim).
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: str = "mps",
        keypoints_cm: Mapping[int, Sequence[float]] | None = None,
        confidence: float = 0.5,
        max_age_seconds: float = 0.2,
        image_size: int = 960,
    ) -> None:
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be nonnegative")
        self.keypoints_cm = dict(MODEL_KEYPOINTS_CM if keypoints_cm is None else keypoints_cm)
        if weights is not None and not self.keypoints_cm:
            raise ValueError("No verified keypoint mapping. Supply the model's documented keypoints_cm.")
        self.device = device
        self.confidence = confidence
        self.max_age_seconds = max_age_seconds
        self.image_size = image_size
        self.scene_id = None
        self.last_keypoints = np.empty((0, 2), dtype=np.float32)
        self.last_keypoint_ids: list[int] = []
        self.fit = CourtFit(False, "awaiting court landmarks", 0.0)
        self.model = None
        if weights is not None:
            # Ultralytics 8.4.146 supports restricted checkpoint reconstruction.
            # Enable before import; never silently fall back to arbitrary pickle.
            os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"
            os.environ.setdefault("YOLO_OFFLINE", "1")
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            from ultralytics import YOLO
            from ultralytics.nn.tasks import SAFE_LOAD, _SafeLoad
            if not SAFE_LOAD or not _SafeLoad.SUPPORTED:
                raise RuntimeError("Set ULTRALYTICS_SAFE_LOAD=1 before importing Ultralytics; torch >=2.6 is required.")
            self.model = YOLO(str(weights), task="pose")

    def reset(self, reason: str = "camera cut") -> None:
        self.fit = CourtFit(False, reason, self.fit.timestamp)
        self.last_keypoints = np.empty((0, 2), dtype=np.float32)
        self.last_keypoint_ids = []

    def update_keypoints(
        self,
        xy: np.ndarray,
        confidence: np.ndarray,
        frame_shape: Sequence[int],
        timestamp: float,
    ) -> CourtFit:
        """Calibrate from one model prediction; exposed for deterministic tests."""
        xy = np.asarray(xy).reshape(-1, 2)
        confidence = np.asarray(confidence).reshape(-1)
        if len(confidence) != len(xy):
            self.reset("keypoint confidence count mismatch")
            return self.fit
        ids = [i for i in sorted(self.keypoints_cm) if i < len(xy) and confidence[i] >= self.confidence]
        self.last_keypoint_ids = ids
        self.last_keypoints = xy[ids].copy()
        targets = np.asarray([self.keypoints_cm[i] for i in ids], dtype=float).reshape(-1, 2)
        targets = targets / [COURT_LENGTH_CM, COURT_WIDTH_CM]
        self.fit = fit_homography(self.last_keypoints, targets, frame_shape, timestamp)
        return self.fit

    def update(self, frame_bgr: np.ndarray, timestamp: float, scene_id=None) -> CourtFit:
        if scene_id != self.scene_id:
            self.reset()
            self.scene_id = scene_id
        if self.model is None:
            raise RuntimeError("CourtMapper needs model weights for update().")
        result = self.model.predict(
            frame_bgr, device=self.device, imgsz=self.image_size,
            conf=0.25, verbose=False, max_det=1,
        )[0]
        keypoints = result.keypoints
        if keypoints is None or len(keypoints) == 0 or keypoints.conf is None:
            self.reset("court landmarks not visible")
            return self.fit
        return self.update_keypoints(
            keypoints.xy[0].detach().cpu().numpy(),
            keypoints.conf[0].detach().cpu().numpy(),
            frame_bgr.shape, timestamp,
        )

    def project(self, points_px: np.ndarray, timestamp: float) -> np.ndarray:
        """NaN marks an unavailable, expired, or off-court position."""
        points = np.asarray(points_px, dtype=float).reshape(-1, 2)
        missing = np.full((len(points), 2), np.nan)
        age = timestamp - self.fit.timestamp
        if (
            not self.fit.valid or self.fit.image_to_court is None
            or not np.isfinite(age) or age < 0 or age > self.max_age_seconds
        ):
            return missing
        result = _transform(points, self.fit.image_to_court)
        valid = np.isfinite(result).all(axis=1) & (result >= 0).all(axis=1) & (result <= 1).all(axis=1)
        result[~valid] = np.nan
        return result

    def annotate_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = frame_bgr.copy()
        for keypoint_id, xy in zip(self.last_keypoint_ids, self.last_keypoints):
            if not np.isfinite(xy).all():
                continue
            point = tuple(np.round(xy).astype(int))
            cv2.circle(result, point, 4, (80, 225, 170), -1, cv2.LINE_AA)
            cv2.putText(result, str(keypoint_id), (point[0] + 4, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 255, 240), 1, cv2.LINE_AA)
        return result


def render_court(
    points_normalized: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
    *,
    width: int = 640,
    trails: Mapping[str, Sequence[Sequence[float]]] | None = None,
    status: str = "NBA court",
) -> np.ndarray:
    """Draw NBA markings and valid positions; colors use OpenCV BGR order."""
    if width < 200:
        raise ValueError("court rendering width must be at least 200 pixels")
    margin = max(15, width // 32)
    play_width = width - 2 * margin
    scale = play_width / COURT_LENGTH_CM
    play_height = int(round(COURT_WIDTH_CM * scale))
    height = play_height + 2 * margin + 32
    image = np.full((height, width, 3), (29, 34, 42), dtype=np.uint8)
    line_color = (203, 210, 215)
    thickness = max(1, width // 500)

    def pixel(x: float, y: float) -> tuple[int, int]:
        return int(round(margin + x * scale)), int(round(margin + y * scale))

    def polyline(coords: Sequence[Sequence[float]], color=line_color, closed=False, thick=thickness):
        pts = np.array([pixel(x, y) for x, y in coords], dtype=np.int32)
        cv2.polylines(image, [pts], closed, color, thick, cv2.LINE_AA)

    cv2.rectangle(image, pixel(0, 0), pixel(COURT_LENGTH_CM, COURT_WIDTH_CM), (55, 67, 78), -1)
    paint_top = (COURT_WIDTH_CM - PAINT_WIDTH_CM) / 2
    for right in (False, True):
        flip = (lambda x: COURT_LENGTH_CM - x) if right else (lambda x: x)
        cv2.rectangle(image, pixel(flip(0), paint_top), pixel(flip(PAINT_LENGTH_CM), paint_top + PAINT_WIDTH_CM), (67, 79, 88), -1)
        polyline([(flip(0), paint_top), (flip(PAINT_LENGTH_CM), paint_top), (flip(PAINT_LENGTH_CM), paint_top + PAINT_WIDTH_CM), (flip(0), paint_top + PAINT_WIDTH_CM)])
        # Circle centered on the free-throw line (the hidden half is dashed).
        for a0, a1 in [(a, a + 0.13) for a in np.arange(np.pi / 2, 3 * np.pi / 2, 0.23)] + [(-np.pi / 2, np.pi / 2)]:
            angles = np.linspace(a0, a1, max(4, int((a1 - a0) * 36)))
            polyline([(flip(PAINT_LENGTH_CM + FREE_THROW_RADIUS_CM * np.cos(a)), COURT_WIDTH_CM / 2 + FREE_THROW_RADIUS_CM * np.sin(a)) for a in angles])
        # Three-point corners join the NBA 23 ft 9 in arc continuously.
        dy = COURT_WIDTH_CM / 2 - CORNER_Y_CM
        theta = float(np.arcsin(dy / ARC_RADIUS_CM))
        join_x = RIM_X_CM + float(np.sqrt(ARC_RADIUS_CM ** 2 - dy ** 2))
        polyline([(flip(0), CORNER_Y_CM), (flip(join_x), CORNER_Y_CM)])
        polyline([(flip(0), COURT_WIDTH_CM - CORNER_Y_CM), (flip(join_x), COURT_WIDTH_CM - CORNER_Y_CM)])
        polyline([(flip(RIM_X_CM + ARC_RADIUS_CM * np.cos(a)), COURT_WIDTH_CM / 2 + ARC_RADIUS_CM * np.sin(a)) for a in np.linspace(-theta, theta, 80)])
        # Restricted semicircle, rim, and backboard.
        polyline([(flip(RIM_X_CM + 122 * np.cos(a)), COURT_WIDTH_CM / 2 + 122 * np.sin(a)) for a in np.linspace(-np.pi / 2, np.pi / 2, 40)])
        cv2.circle(image, pixel(flip(RIM_X_CM), COURT_WIDTH_CM / 2), max(2, round(23 * scale)), (80, 150, 245), thickness, cv2.LINE_AA)
        polyline([(flip(122), COURT_WIDTH_CM / 2 - 91), (flip(122), COURT_WIDTH_CM / 2 + 91)])

    polyline([(0, 0), (COURT_LENGTH_CM, 0), (COURT_LENGTH_CM, COURT_WIDTH_CM), (0, COURT_WIDTH_CM)], closed=True)
    polyline([(COURT_LENGTH_CM / 2, 0), (COURT_LENGTH_CM / 2, COURT_WIDTH_CM)])
    cv2.circle(image, pixel(COURT_LENGTH_CM / 2, COURT_WIDTH_CM / 2), round(183 * scale), line_color, thickness, cv2.LINE_AA)
    if trails:
        for trail in trails.values():
            # Splitting at invalid points prevents joining across missing fits.
            segment = []
            for point in list(trail) + [[float("nan"), float("nan")]]:
                point = np.asarray(point)
                if point.shape == (2,) and np.isfinite(point).all() and (point >= 0).all() and (point <= 1).all():
                    segment.append(point * [COURT_LENGTH_CM, COURT_WIDTH_CM])
                else:
                    if len(segment) > 1:
                        polyline(segment, (122, 130, 132), thick=1)
                    segment = []

    if points_normalized is not None:
        for i, point in enumerate(np.asarray(points_normalized).reshape(-1, 2)):
            if not np.isfinite(point).all() or (point < 0).any() or (point > 1).any():
                continue
            center = pixel(point[0] * COURT_LENGTH_CM, point[1] * COURT_WIDTH_CM)
            color = colors[i] if colors is not None and i < len(colors) else (100, 210, 245)
            cv2.circle(image, center, max(4, width // 90), (20, 24, 29), -1, cv2.LINE_AA)
            cv2.circle(image, center, max(3, width // 110), color, -1, cv2.LINE_AA)
            if labels is not None and i < len(labels):
                cv2.putText(image, str(labels[i]), (center[0] + 7, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, status[:90], (margin, height - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (174, 188, 200), 1, cv2.LINE_AA)
    return image
