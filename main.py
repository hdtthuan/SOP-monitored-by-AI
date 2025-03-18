import argparse
import cv2
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
from ultralytics import YOLO
import mediapipe as mp
import global_variable  # Imported to show usage of global variables, if needed
from modules.action_screwdriver_module import (
    process_frame
)
import global_variable
from modules.hand_detection_v3 import process_frame
from modules.roi_selection import roi_selection_loop


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser(description="ROI selection and hand detection.")
parser.add_argument("--source", type=str, default="./data/Val_video.mp4",
                    help="Path to video file or camera index (default: 0)")
parser.add_argument("--yolo_weight", type=str, default="./models/last_bk0.3.pt",
                    help="Path to video file or camera index (default: 0)")
args = parser.parse_args()

# Determine the video source (camera index or file path)
try:
    source = int(args.source)
except ValueError:
    source = args.source

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Unable to open video source", source)
    exit()

roi_selection_loop(source)

hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )

model = YOLO(args.yolo_weight)

while True:
    ret, frame = cap.read() # TODO: do you see variable frame ?, it can be input, and you can use cv2 draw into this frame inside function.
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    # YOLO Predictions
    results = model(frame, verbose=False, conf=0.3)
    for r in results:
        # Iterate over each detection box in the result
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            # Draw the bounding box (converted to integers)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Optionally, draw the confidence score above the box
            cv2.putText(frame, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # MediaPipe Hand Pose Detection
    results_hands = hands_detector.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Draw hand landmarks and connections on the original frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    output_frame, label, conf = process_frame(frame_copy, results_hands)
    if label:
        print(label, conf)

    # Display the frame with drawn boxes and hand poses
    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()