# server.py - SignLab backend.
# Runs the SAME recognition pipeline as predict.py (same preprocess, same model.pkl,
# same thresholds) inside a background thread, and exposes it to a web UI:
#   /video   MJPEG stream of the annotated camera
#   /events  live state (Server-Sent Events)
#   /api/*   add / delete / clear / speak / practice / correct / settings
#
# Run:  pip install flask   then   python server.py   ->  open http://127.0.0.1:5000
# Close predict.py first (only one program can use the camera).

import datetime
import glob
import json
import os
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utils import FaceTracker, draw_face
from grammar import to_sentence
from preprocess import extract_frame_raw, resample_by_time, sequence_features
from signs import IDLE, display_name

# --------------------------------------------------
# SETTINGS (identical to predict.py - do not change these to "tune" accuracy,
# use the live slider in the UI instead)
# --------------------------------------------------
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
CLASSIFIER_FILE = "model.pkl"
SEQ_DIR = "data/sequences"

WINDOW_SECONDS = 1.5
PREDICT_EVERY = 3
STABLE_WINDOW = 5
STABLE_COUNT = 4
RELEASE_IDLE_COUNT = 2
RELEASE_NO_HAND_SECONDS = 0.8
GREEN = (0, 255, 0)


class Engine:
    def __init__(self):
        saved = joblib.load(CLASSIFIER_FILE)
        self.clf = saved["model"]
        self.words = [str(s) for s in self.clf.classes_ if s != IDLE]

        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face_tracker = FaceTracker(FACE_MODEL_PATH)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise SystemExit("Could not access camera. Close predict.py / other apps using it.")

        self.lock = threading.Lock()
        self.jpeg = None
        self.running = True

        self.sentence = []                 # sign keys, e.g. "THANK_YOU"
        self.corrected = ""
        self.settings = {"conf": 0.50, "use_llm": False}
        self.practice = {"target": None, "success_id": 0, "last_success": None}
        self.added_id = 0                  # bumps whenever a word is auto-added

        self.seeing = (IDLE, 0.0)
        self.top = []
        self.status = "warming"
        self.hands = 0
        self.face_ok = False
        self.last_seq = None               # last (30,131) window, for "teach the model"
        self.fps = 0.0

    # ---------------- sentence helpers (call with lock held) ----------------
    def _glosses(self):
        return [display_name(s) for s in self.sentence]

    def _refresh_text(self):
        self.corrected = to_sentence(self._glosses(), use_llm=False) if self.sentence else ""

    def add_word(self, sign):
        with self.lock:
            if sign not in self.words:
                return False
            self.sentence.append(sign)
            self._refresh_text()
            return True

    def delete_last(self):
        with self.lock:
            if self.sentence:
                self.sentence.pop()
            self._refresh_text()

    def clear(self):
        with self.lock:
            self.sentence.clear()
            self._refresh_text()

    def speak_text(self):
        with self.lock:
            glosses = self._glosses()
            use_llm = self.settings["use_llm"]
        text = to_sentence(glosses, use_llm=use_llm) if glosses else ""   # may take up to 3 s
        with self.lock:
            self.corrected = text
        return text

    def save_sample(self, sign):
        """Store the current 1.5 s window as a new training clip for `sign`."""
        with self.lock:
            seq = None if self.last_seq is None else self.last_seq.copy()
        if seq is None or not seq[:, :126].any():
            return False                                   # no hand in the window
        d = os.path.join(SEQ_DIR, sign)
        os.makedirs(d, exist_ok=True)
        stamp = datetime.date.today().strftime("%Y%m%d")
        n = len(glob.glob(os.path.join(d, f"ui{stamp}_*.npy")))
        np.save(os.path.join(d, f"ui{stamp}_{n:04d}.npy"), seq.astype(np.float32))
        return True

    def state(self):
        with self.lock:
            return {
                "seeing": {"sign": str(self.seeing[0]), "name": display_name(self.seeing[0]),
                           "conf": round(self.seeing[1], 3)},
                "top": self.top,
                "status": self.status,
                "hands": self.hands,
                "face": self.face_ok,
                "fps": round(self.fps, 1),
                "sentence": [{"sign": s, "name": display_name(s)} for s in self.sentence],
                "corrected": self.corrected,
                "practice": {"target": self.practice["target"],
                             "success_id": self.practice["success_id"],
                             "last_success": self.practice["last_success"]},
                "added_id": self.added_id,
                "settings": self.settings,
            }

    # ---------------- main loop (same logic as predict.py) ----------------
    def run(self):
        buffer = deque()
        recent = deque(maxlen=STABLE_WINDOW)
        locked = None
        idle_streak = 0
        last_hand_time = time.time()
        frame_count = 0
        t_prev = time.time()

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            frame_count += 1
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # detection uses the RAW frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hands = self.detector.detect(mp_image).hand_landmarks
            face = self.face_tracker.update(mp_image, w, h)

            draw_face(frame, face)
            for hand in hands:
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, GREEN, -1)
            if hands:
                last_hand_time = now

            buffer.append((now, extract_frame_raw(hands, face)))
            while buffer and now - buffer[0][0] > WINDOW_SECONDS:
                buffer.popleft()
            window_full = buffer[-1][0] - buffer[0][0] >= 0.9 * WINDOW_SECONDS

            if window_full and frame_count % PREDICT_EVERY == 0:
                times = [t for t, _ in buffer]
                frames = [f for _, f in buffer]
                seq = resample_by_time(times, frames)
                probs = self.clf.predict_proba(sequence_features(seq).reshape(1, -1))[0]
                order = np.argsort(probs)[::-1]
                best = int(order[0])
                label, conf = self.clf.classes_[best], float(probs[best])
                recent.append((label, conf))

                top = [{"sign": str(self.clf.classes_[i]), "name": display_name(self.clf.classes_[i]),
                        "conf": round(float(probs[i]), 3)}
                       for i in order if self.clf.classes_[i] != IDLE][:3]

                with self.lock:
                    self.seeing = (label, conf)
                    self.top = top
                    self.last_seq = seq
                    conf_thr = self.settings["conf"]

                idle_streak = idle_streak + 1 if label == IDLE else 0
                if idle_streak >= RELEASE_IDLE_COUNT or now - last_hand_time >= RELEASE_NO_HAND_SECONDS:
                    locked = None

                confident = [l for l, c in recent if c >= conf_thr and l != IDLE]
                if len(recent) == STABLE_WINDOW and confident:
                    word, votes = Counter(confident).most_common(1)[0]
                    if votes >= STABLE_COUNT and word != locked:
                        with self.lock:
                            target = self.practice["target"]
                            if target:                                  # practice mode
                                if word == target:
                                    self.practice["success_id"] += 1
                                    self.practice["last_success"] = target
                                    self.practice["target"] = None
                                    locked = word
                                    recent.clear()
                            else:                                       # translate mode
                                self.sentence.append(str(word))
                                self._refresh_text()
                                self.added_id += 1
                                locked = word
                                recent.clear()

            with self.lock:
                self.hands = len(hands)
                self.face_ok = bool(face[4] > 0)
                if locked:
                    self.status = "added"
                elif not window_full:
                    self.status = "warming"
                elif not hands:
                    self.status = "no_hand"
                else:
                    self.status = "ready"

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.jpeg = buf.tobytes()

        self.cap.release()
        self.detector.close()
        self.face_tracker.close()


engine = Engine()
app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/video")
def video():
    def gen():
        last = None
        while True:
            jpg = engine.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.02)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    def gen():
        while True:
            yield f"data: {json.dumps(engine.state())}\n\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/signs")
def api_signs():
    return jsonify([{"sign": s, "name": display_name(s)} for s in engine.words])


@app.route("/api/add", methods=["POST"])
def api_add():
    return jsonify(ok=engine.add_word(request.json.get("sign", "")))


@app.route("/api/correct", methods=["POST"])
def api_correct():
    """User says 'that was really X': add it AND save the window as training data."""
    sign = request.json.get("sign", "")
    saved = engine.save_sample(sign) if sign in engine.words else False
    return jsonify(ok=engine.add_word(sign), saved=saved)


@app.route("/api/delete", methods=["POST"])
def api_delete():
    engine.delete_last()
    return jsonify(ok=True)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    engine.clear()
    return jsonify(ok=True)


@app.route("/api/sentence", methods=["POST"])
def api_sentence():
    return jsonify(text=engine.speak_text())


@app.route("/api/practice", methods=["POST"])
def api_practice():
    target = request.json.get("target")
    with engine.lock:
        engine.practice["target"] = target if target in engine.words else None
    return jsonify(ok=True)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.json or {}
    with engine.lock:
        if "conf" in data:
            engine.settings["conf"] = min(0.95, max(0.2, float(data["conf"])))
        if "use_llm" in data:
            engine.settings["use_llm"] = bool(data["use_llm"])
    return jsonify(ok=True)


if __name__ == "__main__":
    threading.Thread(target=engine.run, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"SignLab running -> http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
