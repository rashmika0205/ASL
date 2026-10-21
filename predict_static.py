import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utils import FaceTracker, draw_face
from grammar import to_sentence
from preprocess import normalize_landmarks
from signs import display_name

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
CLASSIFIER_FILE = "model_static.pkl"

USE_LLM = False             # True = use the Claude API in grammar.py (needs internet + API key)

IDLE = "IDLE"
SMOOTH_FRAMES = 7           # majority vote over the last N frames
CONF_THRESHOLD = 0.70       # minimum average confidence to count a sign
HOLD_SECONDS = 0.8          # hold a sign this long to add the word
RELEASE_SECONDS = 0.8       # hand down / no sign this long to allow a repeat

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

saved = joblib.load(CLASSIFIER_FILE)
classifier = saved["model"]
uses_face = saved["uses_face"]

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)
face_tracker = FaceTracker(FACE_MODEL_PATH) if uses_face else None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not access camera.")
    exit()

# --------------------------------------------------
# STATE
# --------------------------------------------------

history = deque(maxlen=SMOOTH_FRAMES)    # (label, confidence)
sentence = []                            # recognised words (glosses)
spoken_text = ""                         # grammar-corrected sentence

candidate = None
candidate_since = 0.0
locked_label = None                      # word just added; blocks repeats
idle_since = None


def speak(text):
    if pyttsx3 is None:
        print("pyttsx3 not installed. Run: pip install pyttsx3")
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def put(frame, text, pos, scale=0.7, color=WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence.clear()
        spoken_text = ""
    elif key == ord("b") and sentence:
        sentence.pop()
        spoken_text = ""
    elif key == ord("s") and sentence:
        put(frame, "SPEAKING...", (w // 2 - 100, h // 2), 1.0, YELLOW, 3)
        cv2.imshow("ASL Sign to Text", frame)
        cv2.waitKey(1)
        glosses = [display_name(s) for s in sentence]
        spoken_text = to_sentence(glosses, use_llm=USE_LLM)
        print(f"{' '.join(glosses)}  ->  {spoken_text}")
        speak(spoken_text)

    # ---------------- detection ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    face = np.zeros(5, dtype=np.float32)
    if face_tracker:
        face = face_tracker.update(mp_image, w, h)
        draw_face(frame, face)

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, GREEN, -1)

        row = []
        for lm in hand:
            row += [lm.x, lm.y, lm.z]

        features = normalize_landmarks(
            row, face if uses_face else None).reshape(1, -1)
        probs = classifier.predict_proba(features)[0]
        best = int(np.argmax(probs))
        history.append((classifier.classes_[best], float(probs[best])))
    else:
        history.append((IDLE, 1.0))

    # ---------------- smoothing ----------------
    votes = Counter(label for label, _ in history)
    shown_label = votes.most_common(1)[0][0]
    confs = [c for label, c in history if label == shown_label]
    shown_conf = sum(confs) / len(confs)

    # ---------------- add a word when a sign is held ----------------
    hold = 0.0
    if shown_label != IDLE and shown_conf >= CONF_THRESHOLD:
        idle_since = None
        if shown_label != candidate:
            candidate = shown_label
            candidate_since = now
        hold = min(1.0, (now - candidate_since) / HOLD_SECONDS)
        if hold >= 1.0 and candidate != locked_label:
            sentence.append(candidate)
            locked_label = candidate
            spoken_text = ""
    else:
        candidate = None
        if shown_label == IDLE:
            if idle_since is None:
                idle_since = now
            elif now - idle_since >= RELEASE_SECONDS:
                locked_label = None

    # ---------------- overlays ----------------
    if shown_label == IDLE:
        put(frame, "IDLE", (20, 40), 1.0, WHITE, 2)
    else:
        color = GREEN if shown_conf >= CONF_THRESHOLD else YELLOW
        put(frame, f"{display_name(shown_label)}  {shown_conf:.0%}",
            (20, 40), 1.0, color, 2)

    if candidate and candidate != locked_label:
        cv2.rectangle(frame, (20, 55), (220, 70), WHITE, 1)
        cv2.rectangle(frame, (20, 55), (20 + int(200 * hold), 70), GREEN, -1)
        put(frame, "hold...", (230, 69), 0.5, WHITE, 1)
    elif locked_label:
        put(frame, f"ADDED {display_name(locked_label)} - lower hand to repeat",
            (20, 70), 0.55, YELLOW, 1)

    if uses_face:
        put(frame, "FACE: OK" if face[4] > 0 else "FACE: NOT FOUND", (20, 100),
            0.55, GREEN if face[4] > 0 else RED, 1)

    cv2.rectangle(frame, (0, h - 135), (w, h - 45), (0, 0, 0), -1)
    glosses_line = " ".join(display_name(s) for s in sentence) or "(signs appear here)"
    put(frame, glosses_line[-48:], (15, h - 105), 0.75, WHITE, 2)
    put(frame, spoken_text[-52:] if spoken_text else "(press S for corrected sentence)",
        (15, h - 70), 0.7, GREEN if spoken_text else (150, 150, 150), 2)

    put(frame, "S = correct + speak   B = delete last   C = clear   Q = quit",
        (15, h - 20), 0.5, WHITE, 1)

    cv2.imshow("ASL Sign to Text", frame)

cap.release()
cv2.destroyAllWindows()
detector.close()
if face_tracker:
    face_tracker.close()