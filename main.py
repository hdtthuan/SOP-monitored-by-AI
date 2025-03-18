import argparse
import cv2
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
from ultralytics import YOLO
import mediapipe as mp
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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # # results = model(frame, verbose=False, conf=0.3) # TODO: Here is result of predict yolo
    # code example
    # for r in results:
    #     for box in r.boxes.data.tolist():
    #         x1, y1, x2, y2, conf, cls = box
    #         print(x1, y1, x2, y2, conf, cls)

    # # results_hands = hands_detector.process(frame_rgb) # TODO: Here is result of hand pose
    # code exmaple
    # if results_hands.multi_hand_landmarks:
    #     for hand_landmarks in results_hands.multi_hand_landmarks:
    #         print(hand_landmarks)

    process_frame(frame)

    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()