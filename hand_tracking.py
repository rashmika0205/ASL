import cv2
import mediapipe as mp

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Downloaded hand landmark model will go here
MODEL_PATH = "hand_landmarker.task"

# Create the hand landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Could not access camera")
        break

    frame = cv2.flip(frame, 1)

    # Convert OpenCV image to MediaPipe image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = detector.detect(mp_image)

    # Draw landmarks
    for hand_landmarks in results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Print the first landmark
            print(
                 "Landmark 0:",
                 hand_landmarks[0].x,
                 hand_landmarks[0].y,
                 hand_landmarks[0].z
                )

    cv2.imshow("ASL - MediaPipe Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()