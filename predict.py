import textwrap
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
from preprocess import extract_frame_raw, resample_by_time, sequence_features
from signs import IDLE, display_name

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
CLASSIFIER_FILE = "model.pkl"

USE_LLM = False            # True = S uses the Claude API (internet + key); falls back to rules

WINDOW_SECONDS = 1.5       # must match CLIP_SECONDS in collect_sequences.py
PREDICT_EVERY = 3          # run the model every N frames

STABLE_WINDOW = 5          # look at the last N predictions...
STABLE_COUNT = 4           # ...a word needs this many agreeing ones
CONF_THRESHOLD = 0.50      # each agreeing prediction needs this confidence
RELEASE_IDLE_COUNT = 2     # IDLE predictions in a row that unlock a repeat
RELEASE_NO_HAND_SECONDS = 0.8

WRAP_CHARS = 42            # characters per line of the corrected sentence

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

saved = joblib.load(CLASSIFIER_FILE)
classifier = saved["model"]

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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not access camera.")
    exit()

# --------------------------------------------------
# STATE
# --------------------------------------------------

buffer = deque()                       # (time, raw frame) of the last window
recent = deque(maxlen=STABLE_WINDOW)   # recent (label, confidence)
sentence = []                          # accepted words (glosses)

locked_label = None                    # word just accepted; blocks repeats
idle_streak = 0
last_hand_time = time.time()
frame_count = 0

seeing_label, seeing_conf = IDLE, 0.0

llm_text = ""                          # text from the last S press (when USE_LLM)
llm_for = ()                           # the words that text was made from


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
    frame_count += 1

    glosses = [display_name(s) for s in sentence]

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence.clear()
    elif key == ord("b") and sentence:
        sentence.pop()
    elif key == ord("s") and sentence:
        put(frame, "SPEAKING...", (w // 2 - 100, h // 2), 1.0, YELLOW, 3)
        cv2.imshow("ASL Prediction", frame)
        cv2.waitKey(1)
        text = to_sentence(glosses, use_llm=USE_LLM)
        llm_text, llm_for = text, tuple(sentence)
        print(f"{' '.join(glosses)}  ->  {text}")
        speak(text)

    # ---------------- detection ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hands = detector.detect(mp_image).hand_landmarks
    face = face_tracker.update(mp_image, w, h)

    draw_face(frame, face)
    for hand in hands:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, GREEN, -1)

    if hands:
        last_hand_time = now

    # ---------------- rolling window ----------------
    buffer.append((now, extract_frame_raw(hands, face)))
    while buffer and now - buffer[0][0] > WINDOW_SECONDS:
        buffer.popleft()

    window_full = buffer[-1][0] - buffer[0][0] >= 0.9 * WINDOW_SECONDS

    if window_full and frame_count % PREDICT_EVERY == 0:
        times = [t for t, _ in buffer]
        frames = [f for _, f in buffer]
        seq = resample_by_time(times, frames)

        probs = classifier.predict_proba(sequence_features(seq).reshape(1, -1))[0]
        best = int(np.argmax(probs))
        seeing_label, seeing_conf = classifier.classes_[best], float(probs[best])
        recent.append((seeing_label, seeing_conf))

        # unlock repeats once the hand rests / leaves
        idle_streak = idle_streak + 1 if seeing_label == IDLE else 0
        if (idle_streak >= RELEASE_IDLE_COUNT
                or now - last_hand_time >= RELEASE_NO_HAND_SECONDS):
            locked_label = None

        # accept a word when enough recent predictions agree
        confident = [l for l, c in recent if c >= CONF_THRESHOLD and l != IDLE]
        if len(recent) == STABLE_WINDOW and confident:
            word, votes = Counter(confident).most_common(1)[0]
            if votes >= STABLE_COUNT and word != locked_label:
                sentence.append(word)
                locked_label = word
                recent.clear()

    # ---------------- corrected sentence (conjunctions added) ----------------
    glosses = [display_name(s) for s in sentence]
    if sentence and llm_for == tuple(sentence):
        corrected = llm_text                       # result of the last S press
    elif sentence:
        corrected = to_sentence(glosses, use_llm=False)   # live, offline
    else:
        corrected = ""

    # ---------------- overlays ----------------
    put(frame, f"Seeing: {display_name(seeing_label)} {seeing_conf:.0%}",
        (20, 35), 0.7, WHITE)

    if locked_label:
        put(frame, f"ADDED {display_name(locked_label)} - lower hand to repeat",
            (20, 65), 0.6, YELLOW)
    elif window_full:
        put(frame, "READY", (20, 65), 0.6, GREEN)
    else:
        put(frame, "WARMING UP...", (20, 65), 0.6, YELLOW)

    put(frame, f"HANDS: {len(hands)}", (20, 95), 0.55,
        GREEN if hands else RED)
    put(frame, "FACE: OK" if face[4] > 0 else "FACE: NOT FOUND", (20, 120), 0.55,
        GREEN if face[4] > 0 else RED)

    cv2.rectangle(frame, (0, h - 175), (w, h - 45), (0, 0, 0), -1)

    gloss_line = " ".join(glosses) or "(signs appear here)"
    put(frame, gloss_line[-55:], (15, h - 148), 0.6, WHITE, 1)

    if corrected:
        lines = textwrap.wrap(corrected, WRAP_CHARS)[-2:]
        for i, text_line in enumerate(lines):
            put(frame, text_line, (15, h - 115 + i * 32), 0.75, GREEN, 2)
    else:
        put(frame, "(sentence with conjunctions appears here)",
            (15, h - 115), 0.6, (150, 150, 150), 1)

    put(frame, "S = speak   B = delete last word   C = clear   Q = quit",
        (15, h - 20), 0.55, WHITE, 1)

    cv2.imshow("ASL Prediction", frame)

cap.release()
cv2.destroyAllWindows()
detector.close()
face_tracker.close()