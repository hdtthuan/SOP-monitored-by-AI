import argparse
import cv2
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
from ultralytics import YOLO
import mediapipe as mp
from modules.hand_detection_v2 import process_frame_action
from modules.roi_selection import roi_selection_loop
from modules.SOP_monitoring import SOPMonitoring  # Import SOPMonitoring

# Initialize SOP Monitoring
sop_monitor = SOPMonitoring()

mp_hands = mp.solutions.hands   
mp_drawing = mp.solutions.drawing_utils

# Argument parsing
parser = argparse.ArgumentParser(description="ROI selection and hand detection.")
parser.add_argument(
    "--source",
    type=str,
    default="./data/Val_video.mp4",
    help="Path to video file or camera index (default: 0)",
)
parser.add_argument(
    "--yolo_weight",
    type=str,
    default="./models/last_bk0.3.pt",
    help="Path to YOLO model weights",
)
args = parser.parse_args()

try:
    source = int(args.source)
except ValueError:
    source = args.source

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Unable to open video source", source)
    exit()

# ROI Selection
roi_selection_loop(source)

hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
)

model = YOLO(args.yolo_weight)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for object and hand detection
    action_status = process_frame_action(frame)

    # Validate detected action against SOP sequence
    if action_status:
        if not sop_monitor.validate_action(action_status):
            print("SOP sequence violated. Stopping process.")
            break  # 🔥 Stop main loop immediately if SOP is violated

    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
