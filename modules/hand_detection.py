import cv2
import mediapipe as mp
from ultralytics import YOLO
from roi_selection import rois_crew, roi_object  # Import ROIs đã chọn

# ===== Biến Toàn Cục =====
missing_objects = set()  # Lưu object bị mất
# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Khởi tạo YOLO model
yolo_path = '/home/tuanphan/AI documents/FPT_AI_Semester/VIET DYNAMIC/SOP-monitored-by-AI/models/last_bk0.3.pt'

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

def update_object_roi_status_from_boxes(detection_boxes, roi_object, counts, threshold_frames=5, overlap_threshold=0.3):
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
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



def detect_actions(object_states, hand_states):
    """
    Kiểm tra trạng thái vật thể và bàn tay để xác định hành động.
    
    - "get object [n]": Khi object n biến mất khỏi ROI.
    - "get crew [m]": Khi tay được đưa vào ROI của crew m.
    - "get Tua Vit": Khi object 5 (Tua vít) biến mất khỏi ROI của nó.
    
    Parameters:
    - object_states: Dictionary {object_index: bool} (True nếu object vẫn trong ROI, False nếu mất)
    - hand_states: Set chứa index của crew mà bàn tay được phát hiện.
    
    Returns:
    - List các hành động phát hiện được.
    """
    actions_list = []

    # Kiểm tra object nào đã biến mất
    for obj_id, is_present in object_states.items():
        if not is_present:  # Nếu object không còn trong ROI
            if obj_id == 5:
                actions_list.append("get Tua Vit")  # Tua vít biến mất
            else:
                actions_list.append(f"get object [{obj_id}]")

    # Kiểm tra tay có trong crew nào
    for crew_id in hand_states:
        actions_list.append(f"get crew [{crew_id}]")

    return actions_list


def hand_detection_loop(source):
    """
    Opens the video source and processes each frame to detect hands (using MediaPipe)
    and objects (using YOLO), then draws appropriate action messages.
    """
    global missing_objects
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source", source)
        exit()

    roi_object_counts = {}  # Track số frame liên tiếp mà object xuất hiện
    object_disappear_count = {}  # Đếm số khung hình liên tiếp object biến mất
    object_states = {}  # Dictionary lưu trạng thái của objects (True nếu còn, False nếu mất)
    disappear_threshold = 30  # 🔥 Tăng threshold từ 25 → 30 frame
    overlap_threshold_default = 0.3  # Mức IoU bình thường
    overlap_threshold_tua_vit = 0.2  # 🔥 Tăng mức IoU cho Tua vít (object 5) từ 0.6 → 0.75

    # Initialize MediaPipe hands detector
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )

    print("Hand detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on the current frame
        results = model(frame)
        detection_boxes = []
        for r in results:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                if conf >= 0:
                    detection_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Kiểm tra object ROIs (Sử dụng mức overlap riêng cho Tua vít)
        object_regions = {}
        for i, roi in enumerate(roi_object):
            obj_id = i + 1
            threshold = overlap_threshold_tua_vit if obj_id == 5 else overlap_threshold_default

            detected = False
            for box in detection_boxes:
                if iou(roi, box) >= threshold:  # 🔥 Chỉ xác nhận nếu IoU đủ lớn
                    detected = True
                    break  # Không cần kiểm tra tiếp nếu đã phát hiện

            # Tăng giảm số frame theo trạng thái xuất hiện/mất
            if detected:
                object_states[obj_id] = True
                object_disappear_count[obj_id] = 0
            else:
                object_disappear_count[obj_id] = object_disappear_count.get(obj_id, 0) + 1
                if object_disappear_count[obj_id] >= disappear_threshold:
                    object_states[obj_id] = False

        # Kiểm tra nếu Tua vít (object 5) xuất hiện lại
        if object_states.get(5, False) and 5 in missing_objects:
            missing_objects.remove(5)

        # Process hand landmarks using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands_detector.process(frame_rgb)
        crew_regions = set()
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected = detect_hand_in_rois(hand_landmarks, rois_crew, frame)
                crew_regions.update(detected)

        # Phát hiện hành động dựa trên trạng thái object và bàn tay
        actions_list = detect_actions(object_states, crew_regions)

        # Draw action messages based on detections
        draw_action_text(frame, actions_list)

        # Optionally, draw ROI boxes for visualization
        for i, roi in enumerate(rois_crew):
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'Crew {i + 1}', (roi[0], roi[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, roi in enumerate(roi_object):
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
            cv2.putText(frame, f'Obj {i + 1}', (roi[0], roi[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Hand Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





