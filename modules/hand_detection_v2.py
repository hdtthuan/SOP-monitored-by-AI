import cv2
import mediapipe as mp
from ultralytics import YOLO
from modules.roi_selection import rois_crew, roi_object  # Import ROIs đã chọn
import numpy as np

# ===== Biến Toàn Cục =====
missing_objects = set()  # Lưu object bị mất
ignored_objects = set()  # 🔥 Lưu danh sách object 1-4 không cần detect lại
last_allen_key_state = True  # 🔥 Keeps track of Allen Key state across frames

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Khởi tạo YOLO model
model = YOLO("models/best1.pt")

def iou(
    boxA: tuple[float, float, float, float], boxB: tuple[float, float, float, float]
) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - boxA: Tuple (x1, y1, x2, y2) representing the first bounding box.
    - boxB: Tuple (x1, y1, x2, y2) representing the second bounding box.

    Returns:
    - IoU value (float) ranging from 0 to 1.
    """
    # Compute the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute width and height of intersection
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)

    # Compute the area of intersection
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union area
    unionArea = areaA + areaB - interArea

    # Avoid division by zero
    return interArea / unionArea if unionArea > 0 else 0.0


def update_object_roi_status_from_boxes(detection_boxes, roi_object, counts, threshold_frames=5, overlap_threshold=0.3):
    """
    Checks each YOLO-detected ROI (roi_object) to see if any detection box overlaps sufficiently.

    Parameters:
    - detection_boxes: List of bounding boxes detected in the current frame.
    - roi_object: {object_name: (x1, y1, x2, y2)} dictionary mapping object names to their ROIs.
    - counts: Dictionary tracking the number of consecutive frames where each ROI is detected.
    - threshold_frames: Number of consecutive frames required for an ROI to be considered active.
    - overlap_threshold: Minimum IoU (Intersection over Union) value required to consider an ROI as detected.

    Returns:
    - active_rois: A set containing indices of ROIs that have been active for at least 'threshold_frames' consecutive detections.
    """
    active_rois = set()  # Stores the indices of active ROIs to avoid duplicates

    for object_name, roi in roi_object.items():  # Iterate through each predefined ROI
        detected = False  # Flag to check if ROI is detected in this frame

        for box in detection_boxes:
            if (
                iou(roi, box) >= overlap_threshold
            ):  # Check if overlap is above threshold
                detected = True
                break  # Exit loop early if detected

        # Update the counts dictionary based on detection status
        counts[object_name] = counts.get(object_name, 0) + 1 if detected else 0

        # If ROI has been detected for the required number of frames, mark it as active
        if counts[object_name] >= threshold_frames:
            active_rois.add(object_name)

    return active_rois  # Return the set of active ROI indices


def detect_hand_in_rois(hand_landmarks: mp.solutions.hands.HandLandmark, rois, frame):
    """
    Checks whether specified finger landmarks (index, middle, and ring finger tips)
    are inside the given ROI regions.

    Parameters:
    - hand_landmarks: Hand landmark object detected by MediaPipe (contains x, y coordinates in normalized form).
    - rois: List of tuples (x1, y1, x2, y2) defining regions of interest.
    - frame: The current video frame (used to scale landmarks to pixel coordinates).

    Returns:
    - A set of indices representing ROIs where fingertips are detected.
    """
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Invalid frame dimensions.")

    detected_regions = set()
    fingertips = [8, 12, 16]  # Index, middle, and ring fingertips

    for idx in fingertips:
        if idx >= len(
            hand_landmarks.landmark
        ):  # Check if the index exists in the landmark list
            continue

        lm = hand_landmarks.landmark[idx]
        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

        for i, (x1, y1, x2, y2) in enumerate(rois):
            if x1 < x < x2 and y1 < y < y2:
                detected_regions.add(i + 1)

    return detected_regions


def draw_action_text(frame: cv2.Mat, actions_list) -> None:
    """
    Displays detected actions from detect_actions() on the screen.

    Parameters:
    - frame: The current video frame where text will be displayed.
    - actions_list: A list of action descriptions to be displayed.

    Returns:
    - None (Modifies the frame in place).
    """
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Invalid frame dimensions.")

    if actions_list:
        text = " | ".join(actions_list)
        text_position = (50, 50)  # Default position

        # Adjust text position if needed based on frame size
        if frame.shape[0] < 100:
            text_position = (10, frame.shape[0] - 20)  # Move text to bottom

        cv2.putText(
            frame,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),  # Red color for visibility
            2,
            cv2.LINE_AA,
        )
        return text


def detect_actions(object_states, crew_regions):
    """
    Detects actions based on object and hand states.

    Parameters:
    - object_states: {object_name: True/False} dictionary mapping object_name to their presence status.
    - crew_regions: A set of crew member IDs detected via hand presence.

    Returns:
    - actions_list: A list of strings describing detected actions.
    """
    actions_list = []

    # Process object-related actions
    for object_name, is_present in object_states.items():
        if not is_present and object_name not in ignored_objects:
            # For Allen Key, ensure the action string has square brackets
            if object_name == "Allen Key":
                actions_list.append("get [Allen Key]")
                # Do NOT add Allen Key to ignored_objects so it can be detected again
            else:
                # For other objects, format as "get [object_name]"
                actions_list.append(f"get [{object_name}]")
                ignored_objects.add(object_name)  # Ignore these objects in future detections

    # If Allen Key has been placed back, add "Put [Allen Key]" action
    if object_states.get("Allen Key") == "returned":
        actions_list.append("Put [Allen Key]")
        object_states["Allen Key"] = True  # Reset its state after detecting the action

    # Process hand-related actions (screw detection)
    for crew_id in crew_regions:
        # Format screw action as "get screw [crew_id]"
        actions_list.append(f"get screw [{crew_id}]")

    return actions_list




"""
Performs hand and object detection in a video stream.

Parameters:
- source: Video source (camera index or file path).

Returns:
- None (Displays processed video in a window).
"""
# missing_objects, ignored_objects
global current_action, object_states


object_disappear_count = {}  # {object_name: disappear_count}
object_states = {object_name: True for object_name in roi_object.keys()}
disappear_threshold = 30
overlap_threshold_default = 0.3
overlap_threshold_tua_vit = 0.1
current_action = None

hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
)

# print("Hand detection started. Press 'q' to quit.")


def adjust_gamma(image, gamma=0.5):  # Giảm sáng (gamma < 1)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_frame_action(frame):
    """
    Processes a single frame of video to detect hands and objects.

    Parameters:
    - frame: The current video frame to be processed.

    Returns:
    - action_status: A string describing the detected action.
    """

    # Object detection using YOLO
    frame_low_gamma = adjust_gamma(frame, gamma=0.5)
    results = model(frame_low_gamma, conf=0.2, iou=0.5)
    # results = model(frame, verbose=False, conf=0.3)
    # for r in results:
    #     # Iterate over each detection box in the result
    #     for box in r.boxes.data.tolist():
    #         x1, y1, x2, y2, conf, cls = box
    #         # Draw the bounding box (converted to integers)
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #         # Optionally, draw the confidence score above the box
    #         cv2.putText(frame, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    detection_boxes = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            object_name = model.names[int(cls)]
            if object_name in ignored_objects:
                continue  # 🔥 Không cần YOLO detect lại object 1-4
            if conf >= 0.2:
                detection_boxes.append((int(x1), int(y1), int(x2), int(y2), cls))

    # Update object state
    global last_allen_key_state  # Ensure we modify the global variable

    for object_name, roi in roi_object.items():
        threshold = (
            overlap_threshold_tua_vit
            if object_name == "Allen Key"
            else overlap_threshold_default
        )

        detected = any(
            iou(roi, (x1, y1, x2, y2)) >= threshold
            for x1, y1, x2, y2, cls in detection_boxes
        )

        if detected:
            object_states[object_name] = True
            object_disappear_count[object_name] = 0

            if object_name == "Allen Key" and not last_allen_key_state:
                object_states["Allen Key"] = "returned"  # 🔥 Mark Allen Key as returned
                last_allen_key_state = True  # Update global tracking state

        else:
            object_disappear_count[object_name] = object_disappear_count.get(object_name, 0) + 1
            if object_disappear_count[object_name] >= disappear_threshold:
                object_states[object_name] = False
                if object_name == "Allen Key":
                    last_allen_key_state = False  # 🔥 Update tracking state when Allen Key disappears


    # Draw object detection results
    for object_name, roi in roi_object.items():
        print("c1", object_name, object_states)
        color = (0, 0, 255) if object_states[object_name] else (100, 100, 100)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{object_name}",
            (roi[0], roi[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    global current_action
    # Hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands_detector.process(frame_rgb)
    crew_regions = set()

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected = detect_hand_in_rois(hand_landmarks, rois_crew, frame)
            crew_regions.update(detected)

    # Detect actions based on object and hand states
    actions = detect_actions(object_states, crew_regions)
    new_action = actions[0] if actions else None

    if new_action and new_action != current_action:
        # print(f"Detected Action: {new_action}")
        current_action = new_action

    # Display detected action
    action_status = draw_action_text(frame, [current_action] if current_action else [])

    return action_status
