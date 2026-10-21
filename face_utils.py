import time

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceTracker:
    """
    Finds the face with MediaPipe's face detector and returns 5 raw values:

        [face_cx, face_cy, face_w, face_h, face_found]

    All position/size values are fractions of the frame (0-1).
    If the face is hidden for a moment (e.g. your hand covers it), the last
    known position is reused for `hold_seconds`, since your face barely moves.
    """

    def __init__(self, model_path="blaze_face_short_range.tflite",
                 hold_seconds=2.0):
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.hold_seconds = hold_seconds
        self.last = None
        self.last_time = 0.0

    def update(self, mp_image, frame_w, frame_h):
        result = self.detector.detect(mp_image)

        if result.detections:
            # use the biggest face if several are visible
            det = max(
                result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            b = det.bounding_box
            self.last = np.array(
                [
                    (b.origin_x + b.width / 2) / frame_w,
                    (b.origin_y + b.height / 2) / frame_h,
                    b.width / frame_w,
                    b.height / frame_h,
                    1.0,
                ],
                dtype=np.float32,
            )
            self.last_time = time.time()
            return self.last.copy()

        if self.last is not None and time.time() - self.last_time <= self.hold_seconds:
            return self.last.copy()

        return np.zeros(5, dtype=np.float32)

    def close(self):
        self.detector.close()


def draw_face(frame, face, color=(255, 200, 0)):
    """Draw the face box (if a face is known)."""
    if face[4] <= 0:
        return
    h, w = frame.shape[:2]
    cx, cy, fw, fh = face[:4]
    x1, y1 = int((cx - fw / 2) * w), int((cy - fh / 2) * h)
    x2, y2 = int((cx + fw / 2) * w), int((cy + fh / 2) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)