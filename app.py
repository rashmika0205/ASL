import base64, time, uuid
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utils import FaceTracker
from grammar import to_sentence
from preprocess import extract_frame_raw, resample_by_time, sequence_features
from signs import IDLE, display_name

# ---- same settings as predict.py ----
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
CLASSIFIER_FILE = "model.pkl"
USE_LLM = False
WINDOW_SECONDS = 1.5
STABLE_WINDOW = 5
STABLE_COUNT = 4
CONF_THRESHOLD = 0.50
RELEASE_IDLE_COUNT = 2
RELEASE_NO_HAND_SECONDS = 0.8

app = Flask(__name__, static_folder="static", static_url_path="")
classifier = joblib.load(CLASSIFIER_FILE)["model"]

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)
face_tracker = FaceTracker(FACE_MODEL_PATH)


class Session:
    def __init__(self):
        self.buffer = deque()
        self.recent = deque(maxlen=STABLE_WINDOW)
        self.locked = None
        self.idle_streak = 0
        self.last_hand = time.time()


sessions = {}


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.post("/api/predict")
def predict():
    d = request.get_json()
    s = sessions.setdefault(d.get("sid") or str(uuid.uuid4()), Session())
    img = base64.b64decode(d["frame"].split(",")[1])
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)                      # same mirror as predict.py
    h, w = frame.shape[:2]
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hands = detector.detect(mp_image).hand_landmarks
    face = face_tracker.update(mp_image, w, h)
    if hands:
        s.last_hand = now

    s.buffer.append((now, extract_frame_raw(hands, face)))
    while s.buffer and now - s.buffer[0][0] > WINDOW_SECONDS:
        s.buffer.popleft()

    span = s.buffer[-1][0] - s.buffer[0][0]
    if span < 0.9 * WINDOW_SECONDS:
        return jsonify(hand=True, ready=False, progress=min(1.0, span / WINDOW_SECONDS))

    seq = resample_by_time([t for t, _ in s.buffer], [f for _, f in s.buffer])
    probs = classifier.predict_proba(sequence_features(seq).reshape(1, -1))[0]
    order = np.argsort(probs)[::-1]
    label, conf = classifier.classes_[order[0]], float(probs[order[0]])
    s.recent.append((label, conf))

    # unlock repeats once the hand rests / leaves
    s.idle_streak = s.idle_streak + 1 if label == IDLE else 0
    if s.idle_streak >= RELEASE_IDLE_COUNT or now - s.last_hand >= RELEASE_NO_HAND_SECONDS:
        s.locked = None

    # accept a word when enough recent predictions agree
    accepted = None
    confident = [l for l, c in s.recent if c >= CONF_THRESHOLD and l != IDLE]
    if len(s.recent) == STABLE_WINDOW and confident:
        word, votes = Counter(confident).most_common(1)[0]
        if votes >= STABLE_COUNT and word != s.locked:
            accepted, s.locked = str(word), word
            s.recent.clear()

    return jsonify(hand=True, ready=True, word=str(label), conf=conf, accepted=accepted,
                   top3=[{"word": str(classifier.classes_[i]), "conf": float(probs[i])}
                         for i in order[:3]])


def gloss(w):
    try:
        return display_name(w)
    except Exception:                      # manually typed word not in signs.py
        return w.replace("_", " ").upper()


@app.post("/api/sentence")
def sentence():
    words = [gloss(w) for w in request.get_json().get("words", []) if w != IDLE]
    return jsonify(text=to_sentence(words, use_llm=USE_LLM))


if __name__ == "__main__":
    app.run(debug=False, port=5000)