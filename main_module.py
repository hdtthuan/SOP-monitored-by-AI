import argparse
import cv2
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
from ultralytics import YOLO
import mediapipe as mp

from modules.roi_selection import roi_selection_loop, ROISelector
from modules.SOP_monitoring_v2 import SOPMonitoring  # Import SOPMonitoring
 
# Initialize SOP Monitoring
sop_monitor = SOPMonitoring()

mp_hands = mp.solutions.hands   
mp_drawing = mp.solutions.drawing_utils

# Argument parsing
parser = argparse.ArgumentParser(description="ROI selection and hand detection.")
parser.add_argument(
    "--source",
    type=str,
    default=r"data\wrong_video\WIN_20250326_09_38_37_Pro.mp4",
    help="Path to video file or camera index (default: 0)",
)
parser.add_argument(
    "--yolo_weight",
    type=str,
    default="./models/best_v4.pt",
    help="Path to YOLO model weights",
)
args = parser.parse_args()
yolo_weight_path = args.yolo_weight
roi_selector = ROISelector("models/best-test.pt")

try:
    source = int(args.source)
except ValueError:
    source = args.source

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Unable to open video source", source)
    exit()

# ROI Selection

roi_selection_loop(source, roi_selector)

from modules.hand_detection_v2 import HandDetector
hand_detector = HandDetector(yolo_weight_path)
from modules.hand_detection_v2 import process_frame_action # tuan ngu
hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3
)

# model = YOLO(args.yolo_weight)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    frame_clone = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for object and hand detection
    action_status = process_frame_action(frame, hand_detector)
    
    if action_status:
        if not sop_monitor.validate_action(action_status):
            # Hide the "Current Action" display by not showing it during error
            frame = frame_clone
            expected = sop_monitor.get_expected_action()
            error_line1 = f"ERROR: expected action is: {expected} but current action is: {action_status}"
            error_line2 = "SOP violated! Press 'C' to retry or 'Q' to quit."

            # Draw error messages on the frame
            cv2.putText(
                frame,
                error_line1,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255), 2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                error_line2,
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255), 2,
                cv2.LINE_AA
            )

            # Show frame with error
            cv2.imshow("Hand Detection", frame)

            # Pause and wait for 'c' or 'q'
            break_process = False
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    print("[INFO] Retrying detection from beginning...")
                    # sop_monitor.reset_monitoring()
                    break_process = True
                    break
                elif key == ord("q"):
                    print("[INFO] Exiting monitoring.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(3)
            if break_process:
                break
            continue

    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit(3)

cap.release()
cv2.destroyAllWindows()
