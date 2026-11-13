"""Seeing: a camera, and hand landmarks from MediaPipe.

Needs:  pip install 'eddy-ai[vision]'
The gesture logic itself lives in eddy.gestures and needs none of this.
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("seeing needs: pip install 'eddy-ai[vision]'") from exc
        self._cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")

    def frames(self) -> Iterator:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self.capture.release()


HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/1/hand_landmarker.task")


def default_hand_model() -> Optional[str]:
    for path in (os.environ.get("EDDY_HAND_MODEL"), os.path.expanduser("~/.eddy/hand_landmarker.task"),
                 "models/hand_landmarker.task"):
        if path and os.path.isfile(path):
            return path
    return None


class HandTracker:
    """Find one hand in a BGR frame and return its 21 landmarks as [x, y, z], in units of image height."""

    def __init__(self, min_confidence: float = 0.5, model_path: Optional[str] = None):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("hand signs need: pip install 'eddy-ai[vision]'") from exc
        self._cv2, self._mp = cv2, mp
        self._legacy = None
        if hasattr(mp, "solutions"):            # MediaPipe before 1.0 ships the hand model inside the package
            self._legacy = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                                    min_detection_confidence=min_confidence,
                                                    min_tracking_confidence=min_confidence)
            return
        model_path = model_path or default_hand_model()
        if not model_path:
            raise RuntimeError("no hand model found. Download it once with:\n  curl -fsSL -o ~/.eddy/hand_landmarker.task "
                               + HAND_MODEL_URL)
        vision = mp.tasks.vision
        self._tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path), num_hands=1,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence))

    def landmarks(self, frame) -> Optional[List[List[float]]]:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            result = self._legacy.process(rgb)
            hands = [h.landmark for h in (result.multi_hand_landmarks or [])]
        else:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            hands = self._tracker.detect(image).hand_landmarks
        if not hands:
            return None
        aspect = frame.shape[1] / frame.shape[0]     # same unit on both axes, so shapes are not stretched
        return [[p.x * aspect, p.y, p.z] for p in hands[0]]

    def close(self) -> None:
        (self._legacy or self._tracker).close()
