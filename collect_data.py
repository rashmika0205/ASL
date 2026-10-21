import cv2
import mediapipe as mp
import csv
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utils import FaceTracker, draw_face

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_FILE = "data/landmarks.csv"
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"

FIRST_COUNTDOWN = 3         # seconds to get ready for the very first sign
TRANSITION_COUNTDOWN = 5    # seconds to change from one sign to the next
RESUME_COUNTDOWN = 3        # seconds after un-pausing
MIN_SAMPLE_GAP = 0.05       # seconds between saved samples
IDLE_GAP = 0.1              # seconds between auto IDLE samples (no hand found)
MAX_AUTO_IDLE = 1000        # stop adding auto IDLE rows after this many

# X = IDLE with your hand VISIBLE but in the wrong place / not signing
LABEL_KEYS = {
    ord("i"): "I",
    ord("y"): "YES",
    ord("n"): "NO",
    ord("w"): "WATER",
    ord("x"): "IDLE",
}

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# --------------------------------------------------
# MODELS
# --------------------------------------------------

if not os.path.exists(FACE_MODEL_PATH):
    raise SystemExit(
        f"{FACE_MODEL_PATH} not found. Download it into this folder first "
        "(see the PowerShell command in the chat)."
    )

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(hand_options)
face_tracker = FaceTracker(FACE_MODEL_PATH)

# --------------------------------------------------
# CSV SETUP
# --------------------------------------------------

HEADER = []
for i in range(21):
    HEADER += [f"x{i}", f"y{i}", f"z{i}"]
HEADER += ["face_cx", "face_cy", "face_w", "face_h", "face_found", "label"]

os.makedirs("data", exist_ok=True)
file_exists = os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0

if file_exists:
    with open(DATA_FILE, newline="") as f:
        existing_header = next(csv.reader(f), None)
    if existing_header != HEADER:
        raise SystemExit(
            f"{DATA_FILE} has an old format (no face columns). Rename it to "
            "landmarks_old.csv and run this again."
        )

file = open(DATA_FILE, "a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(HEADER)

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera.")
    exit()

# --------------------------------------------------
# STATE
#   IDLE       -> nothing selected yet
#   COUNTDOWN  -> NOT recording, time to get ready / change sign
#   COLLECTING -> recording samples for current_label
#   PAUSED     -> NOT recording, manual pause
# --------------------------------------------------

state = "IDLE"
current_label = None
previous_label = None
countdown_start = 0.0
countdown_length = 0
last_sample_time = 0.0
last_idle_time = 0.0

counts = {label: 0 for label in LABEL_KEYS.values()}
auto_idle_count = 0
total_samples = 0


def start_countdown(new_label, from_label, length):
    """Stop recording and give the user `length` seconds to get ready."""
    global state, current_label, previous_label
    global countdown_start, countdown_length
    previous_label = from_label
    current_label = new_label
    countdown_start = time.time()
    countdown_length = length
    state = "COUNTDOWN"
    print()
    print("------------------------------")
    if from_label and from_label != new_label:
        print(f"CHANGE SIGN: {from_label} -> {new_label}  ({length}s)")
    else:
        print(f"GET READY: {new_label}  ({length}s)")
    print("Recording is OFF during countdown.")
    print("------------------------------")


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

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # ---------------- detection (always on, so you can see it) ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = detector.detect(mp_image)
    hand = results.hand_landmarks[0] if results.hand_landmarks else None

    face = face_tracker.update(mp_image, w, h)
    face_values = [float(v) for v in face]

    draw_face(frame, face)

    if hand:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, GREEN, -1)

    # ---------------- keys ----------------
    if key in LABEL_KEYS:
        new_label = LABEL_KEYS[key]

        if state == "IDLE":
            start_countdown(new_label, None, FIRST_COUNTDOWN)

        elif state == "COLLECTING" and new_label != current_label:
            start_countdown(new_label, current_label, TRANSITION_COUNTDOWN)

        elif state == "COUNTDOWN" and new_label != current_label:
            start_countdown(new_label, previous_label, TRANSITION_COUNTDOWN)

        elif state == "PAUSED":
            start_countdown(new_label, current_label, TRANSITION_COUNTDOWN)

    elif key == ord("p"):
        if state == "COLLECTING":
            state = "PAUSED"
            print("\nPAUSED - recording is OFF")
        elif state == "PAUSED":
            start_countdown(current_label, None, RESUME_COUNTDOWN)

    # ---------------- state logic ----------------
    if state == "COUNTDOWN":
        remaining = countdown_length - int(time.time() - countdown_start)

        if remaining <= 0:
            state = "COLLECTING"
            print(f"RECORDING: {current_label}")
        else:
            cv2.rectangle(frame, (10, 10), (w - 10, 200), (0, 0, 0), -1)

            if previous_label and previous_label != current_label:
                title = f"CHANGE SIGN: {previous_label} -> {current_label}"
            else:
                title = f"GET READY: {current_label}"

            put(frame, title, (30, 60), 1.0, YELLOW, 3)
            put(frame, str(remaining), (w // 2 - 30, 150), 2.5, YELLOW, 5)
            put(frame, "NOT RECORDING", (w - 260, 185), 0.7, RED)

    elif state == "COLLECTING":
        if hand and time.time() - last_sample_time >= MIN_SAMPLE_GAP:
            row = []
            for lm in hand:
                row += [lm.x, lm.y, lm.z]
            row += face_values
            row.append(current_label)

            writer.writerow(row)
            file.flush()
            last_sample_time = time.time()

            counts[current_label] += 1
            total_samples += 1

        elif (hand is None
              and auto_idle_count < MAX_AUTO_IDLE
              and time.time() - last_idle_time >= IDLE_GAP):
            # No hand found -> IDLE sample (63 zeros = "no landmarks")
            row = [0.0] * 63 + face_values + ["IDLE"]

            writer.writerow(row)
            file.flush()
            last_idle_time = time.time()

            counts["IDLE"] += 1
            auto_idle_count += 1
            total_samples += 1

    # ---------------- overlays ----------------
    if state == "IDLE":
        status, color = "NO SIGN SELECTED", WHITE
    elif state == "COLLECTING":
        if hand:
            status, color = f"RECORDING: {current_label}", GREEN
        else:
            status, color = "RECORDING: IDLE (no hand)", YELLOW
    elif state == "PAUSED":
        status, color = "PAUSED - NOT RECORDING", YELLOW
    else:
        status, color = "", WHITE

    if status:
        put(frame, status, (20, 40), 0.9, color)

    if state != "COUNTDOWN":
        put(frame, "HAND: OK" if hand else "HAND: NOT FOUND",
            (20, 75), 0.7, GREEN if hand else RED)
        put(frame, "FACE: OK" if face[4] > 0 else "FACE: NOT FOUND",
            (20, 103), 0.7, GREEN if face[4] > 0 else RED)

    y = 250 if state == "COUNTDOWN" else 140
    put(frame, f"Total: {total_samples}", (20, y), 0.7)
    y += 30
    for label, count in counts.items():
        marker = "<" if label == current_label and state == "COLLECTING" else ""
        put(frame, f"{label}: {count} {marker}", (20, y), 0.65)
        y += 28

    put(frame, "I=I  Y=YES  N=NO  W=WATER  X=IDLE(hand visible)",
        (20, h - 50), 0.55)
    put(frame, "P=PAUSE/RESUME  Q=QUIT", (20, h - 20), 0.6)

    cv2.imshow("ASL Data Collection", frame)

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

file.close()
cap.release()
cv2.destroyAllWindows()
detector.close()
face_tracker.close()

print("\n==============================")
print("Data collection finished.")
print("==============================")
print(f"Total samples: {total_samples}")
for label, count in counts.items():
    print(f"{label}: {count}")