import cv2
import mediapipe as mp
from ultralytics import YOLO
from roi_selection import rois_crew, roi_object  # Import ROIs đã chọn

# ===== Biến Toàn Cục =====
missing_objects = set()  # Lưu object bị mất
ignored_objects = set()  # 🔥 Lưu danh sách object 1-4 không cần detect lại

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Khởi tạo YOLO model
yolo_path = "/home/tuanphan/AI documents/FPT_AI_Semester/VIET DYNAMIC/SOP-monitored-by-AI/models/last_bk0.3.pt"

model = YOLO(yolo_path)


def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is defined as (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea


def update_object_roi_status_from_boxes(
    detection_boxes, roi_object, counts, threshold_frames=5, overlap_threshold=0.3
):
    """
    Checks each YOLO-detected ROI (roi_object) to see if any detection box overlaps sufficiently.
    Updates a counts dictionary to track consecutive frames with detection.
    """
    active_rois = set()
    for i, roi in enumerate(roi_object):
        detected = False
        for box in detection_boxes:
            if iou(roi, box) >= overlap_threshold:
                detected = True
                break
        counts[i] = counts.get(i, 0) + 1 if detected else 0
        if counts[i] >= threshold_frames:
            active_rois.add(i + 1)
    return active_rois


def detect_hand_in_rois(hand_landmarks, rois, frame):
    """
    Checks whether specified finger landmarks (index, middle, and ring finger tips)
    are inside the given ROI regions (for roi_crew).
    """
    detected_regions = set()
    # Only check fingertips: index (8), middle (12), and ring (16)
    for idx in [8, 12, 16]:
        lm = hand_landmarks.landmark[idx]
        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi
            if x1 < x < x2 and y1 < y < y2:
                detected_regions.add(i + 1)
    return detected_regions


def draw_action_text(frame, actions_list):
    """
    Hiển thị kết quả phát hiện từ detect_actions() trên màn hình.
    """
    if actions_list:
        text = " | ".join(actions_list)
        cv2.putText(
            frame,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def detect_actions(object_states, hand_states):
    actions_list = []
    for obj_id, is_present in object_states.items():
        if not is_present and obj_id not in ignored_objects:  # 🔥 Chỉ detect 1 lần
            if obj_id == 5:
                actions_list.append("get Tua Vit")
            else:
                actions_list.append(f"get object [{obj_id}]")
                ignored_objects.add(obj_id)  # 🔥 Không cần YOLO detect lại object này
    for crew_id in hand_states:
        actions_list.append(f"get crew [{crew_id}]")
    return actions_list


def hand_detection_loop(source):
    global missing_objects, ignored_objects
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source", source)
        exit()

    object_disappear_count = {}
    object_states = {
        i + 1: True for i in range(len(roi_object))
    }  # 🔥 Đảm bảo tất cả object ban đầu được hiển thị
    disappear_threshold = 30
    overlap_threshold_default = 0.3
    overlap_threshold_tua_vit = 0.1
    current_action = None

    hands_detector = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
    )

    print("Hand detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)
        detection_boxes = []
        for r in results:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                cls = int(cls)
                if cls in ignored_objects:
                    continue  # 🔥 Không cần YOLO detect lại object 1-4
                if conf >= 0.3:
                    detection_boxes.append((int(x1), int(y1), int(x2), int(y2), cls))

        for i, roi in enumerate(roi_object):
            obj_id = i + 1
            threshold = (
                overlap_threshold_tua_vit if obj_id == 5 else overlap_threshold_default
            )
            detected = any(
                iou(roi, (x1, y1, x2, y2)) >= threshold
                for x1, y1, x2, y2, cls in detection_boxes
            )
            if detected:
                object_states[obj_id] = True
                object_disappear_count[obj_id] = 0
            else:
                disappear_limit = (
                    disappear_threshold if obj_id == 5 else disappear_threshold
                )
                object_disappear_count[obj_id] = (
                    object_disappear_count.get(obj_id, 0) + 1
                )
                if object_disappear_count[obj_id] >= disappear_limit:
                    object_states[obj_id] = False

        for i, roi in enumerate(roi_object):
            x1, y1, x2, y2 = roi
            color = (0, 0, 255) if object_states[i + 1] else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Obj {i + 1}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        for i, roi in enumerate(rois_crew):
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Crew {i + 1}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands_detector.process(frame_rgb)
        crew_regions = set()
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                detected = detect_hand_in_rois(hand_landmarks, rois_crew, frame)
                crew_regions.update(detected)

        new_action = detect_actions(object_states, crew_regions)
        if isinstance(new_action, list):
            new_action = new_action[0] if new_action else None
        if new_action and new_action != current_action:
            print(f"Detected Action: {new_action}")
            current_action = new_action

        draw_action_text(frame, [current_action] if current_action else [])
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
