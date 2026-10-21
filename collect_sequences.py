import glob
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utils import FaceTracker, draw_face
from preprocess import extract_frame_raw, resample_by_time
from signs import IDLE, SIGNS, TWO_HAND_SIGNS, display_name

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
SEQ_DIR = "data/sequences"

TARGET_CLIPS = 80            # clips to collect per sign
CLIP_SECONDS = 1.5           # length of each clip
FIRST_READY_SECONDS = 3.0    # countdown before the first clip
REST_SECONDS = 1.2           # rest between clips
MIN_HAND_RATIO = 0.6         # share of frames that must show the needed hands
MIN_FRAMES = 10              # clip rejected if the camera gave fewer frames

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

SESSION = datetime.now().strftime("%Y%m%d-%H%M%S")   # no underscore on purpose

# --------------------------------------------------
# MODELS + CAMERA
# --------------------------------------------------

for path in (HAND_MODEL_PATH, FACE_MODEL_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found in this folder.")

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
#   SELECT : choose a sign with A / D
#   READY  : countdown (not recording)
#   RECORD : recording one clip
# --------------------------------------------------

os.makedirs(SEQ_DIR, exist_ok=True)

counts = {s: len(glob.glob(os.path.join(SEQ_DIR, s, "*.npy"))) for s in SIGNS}
saved_stack = []             # (sign, path) of clips saved this session, for undo

sel = 0
phase = "SELECT"
ready_until = 0.0
rec_start = 0.0
rec_times, rec_frames, rec_hand_counts = [], [], []

message = ""
message_until = 0.0


def set_message(text, seconds=3.0):
    global message, message_until
    message = text
    message_until = time.time() + seconds
    print(text)


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

    # ---------------- detection (always on) ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hands = detector.detect(mp_image).hand_landmarks
    face = face_tracker.update(mp_image, w, h)

    draw_face(frame, face)
    for hand in hands:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, GREEN, -1)

    # ---------------- keys ----------------
    if phase == "SELECT":
        if key == ord("a"):
            sel = (sel - 1) % len(SIGNS)
        elif key == ord("d"):
            sel = (sel + 1) % len(SIGNS)
        elif key == 32:                                    # SPACE = start
            phase = "READY"
            ready_until = now + FIRST_READY_SECONDS
            print(f"\nGET READY: {display_name(SIGNS[sel])}")
    elif key == 32:                                        # SPACE = stop
        phase = "SELECT"
        set_message("Stopped.")

    if key == ord("u") and phase != "RECORD":              # U = undo last clip
        if saved_stack:
            last_sign, last_path = saved_stack.pop()
            if os.path.exists(last_path):
                os.remove(last_path)
                counts[last_sign] -= 1
                set_message(f"Undid last clip of {display_name(last_sign)}.")
        else:
            set_message("Nothing to undo in this session.")

    sign = SIGNS[sel]
    need_hands = 2 if sign in TWO_HAND_SIGNS else 1

    # ---------------- READY -> RECORD ----------------
    if phase == "READY" and now >= ready_until:
        phase = "RECORD"
        rec_start = now
        rec_times, rec_frames, rec_hand_counts = [], [], []

    # ---------------- RECORD ----------------
    if phase == "RECORD":
        rec_times.append(now)
        rec_frames.append(extract_frame_raw(hands, face))
        rec_hand_counts.append(len(hands))

        if now - rec_start >= CLIP_SECONDS:
            n = len(rec_frames)
            good_frames = sum(1 for c in rec_hand_counts if c >= need_hands)

            reason = None
            if n < MIN_FRAMES:
                reason = f"only {n} frames recorded (camera too slow?)"
            elif sign != IDLE and good_frames / n < MIN_HAND_RATIO:
                reason = (f"{need_hands} hand(s) visible in only "
                          f"{good_frames}/{n} frames")

            if reason:
                set_message(f"REJECTED: {reason}")
            else:
                clip = resample_by_time(rec_times, rec_frames)
                folder = os.path.join(SEQ_DIR, sign)
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(
                    folder, f"{SESSION}_{int(time.time() * 1000)}.npy")
                np.save(path, clip)
                saved_stack.append((sign, path))
                counts[sign] += 1
                set_message(f"Saved {display_name(sign)}: {counts[sign]}/{TARGET_CLIPS}",
                            1.0)

            if counts[sign] >= TARGET_CLIPS:
                phase = "SELECT"
                set_message(f"{display_name(sign)} reached {TARGET_CLIPS} clips. "
                            "Pick the next sign with D.", 4.0)
            else:
                phase = "READY"
                ready_until = time.time() + REST_SECONDS

    # ---------------- overlays ----------------
    cv2.rectangle(frame, (10, 10), (w - 10, 115), (0, 0, 0), -1)

    title = f"{display_name(sign)}  ({counts[sign]}/{TARGET_CLIPS})"
    if phase == "SELECT":
        put(frame, f"SELECTED: {title}", (25, 45), 0.9, YELLOW)
        put(frame, "SPACE = start", (25, 80), 0.7)
    elif phase == "READY":
        remaining = max(0.0, ready_until - now)
        put(frame, f"GET READY: {display_name(sign)}", (25, 50), 1.0, YELLOW, 3)
        put(frame, f"{remaining:.1f}s", (w - 160, 90), 1.2, YELLOW, 3)
    else:
        put(frame, f"DO THE SIGN NOW: {display_name(sign)}", (25, 50), 1.0, RED, 3)
        fraction = min(1.0, (now - rec_start) / CLIP_SECONDS)
        cv2.rectangle(frame, (25, 80), (w - 25, 100), WHITE, 2)
        cv2.rectangle(frame, (25, 80),
                      (25 + int((w - 50) * fraction), 100), RED, -1)

    hands_ok = len(hands) >= need_hands or sign == IDLE
    put(frame, f"HANDS: {len(hands)} (need {need_hands if sign != IDLE else 0})",
        (20, 145), 0.6, GREEN if hands_ok else RED)
    put(frame, "FACE: OK" if face[4] > 0 else "FACE: NOT FOUND",
        (20, 170), 0.6, GREEN if face[4] > 0 else RED)

    if sign == IDLE:
        put(frame, "IDLE: rest hand / move randomly / leave frame",
            (20, 200), 0.5, YELLOW, 1)

    # sign list in two columns
    for i, s in enumerate(SIGNS):
        col, row = divmod(i, 11)
        x = w - 310 + col * 155
        y = 145 + row * 22
        done = counts[s] >= TARGET_CLIPS
        color = GREEN if done else WHITE
        marker = ">" if i == sel else " "
        put(frame, f"{marker}{display_name(s)}: {counts[s]}", (x, y), 0.45, color, 1)

    if time.time() < message_until:
        put(frame, message, (20, h - 80), 0.6, YELLOW)

    put(frame, "A/D = prev/next sign   SPACE = start/stop", (20, h - 50), 0.55)
    put(frame, "U = undo last clip      Q = quit", (20, h - 22), 0.55)

    cv2.imshow("ASL Sequence Collection", frame)

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()
face_tracker.close()

print("\n==============================")
print("Clips per sign:")
for s in SIGNS:
    print(f"  {display_name(s)}: {counts[s]}")
print("==============================")