import cv2
from ultralytics import YOLO

# ===== Biến Toàn Cục =====
rois_crew = []  # Lưu ROIs được chọn thủ công
roi_object = []  # Lưu ROIs của YOLO
selecting = False  # Đánh dấu trạng thái đang chọn ROI
start_point = None  # Điểm bắt đầu khi vẽ ROI
frame = None  # Frame video hiện tại

# Khởi tạo YOLO model
model = YOLO("../models/best.pt")

def draw_roi(event, x, y, flags, param):
    """
    Handles mouse events for manual ROI selection.
    - Press left mouse button to start drawing.
    - Drag the mouse to adjust ROI size.
    - Release the mouse button to finalize the ROI.
    """
    global selecting, start_point, frame, rois_crew
    if event == cv2.EVENT_LBUTTONDOWN:  # Start ROI selection
        start_point = (x, y)
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:  # Update ROI rectangle during drag
        frame_copy = frame.copy()
        # Draw previously selected manual ROIs
        for i, roi in enumerate(rois_crew):
            cv2.rectangle(frame_copy, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            cv2.putText(frame_copy, f'Crew {i + 1}', (roi[0], roi[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw the ROI being currently selected
        cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:  # Finalize the ROI selection
        rois_crew.append((start_point[0], start_point[1], x, y))
        selecting = False


def yolo_detect_initial_rois(frame, model, label_accept=[]):
    """
    Uses YOLO to detect objects in the current frame and returns their bounding boxes as ROIs.
    """
    results = model(frame)
    new_rois = []
    # YOLOv8 returns a list of Results objects (one per image)
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            print(x1, y1, x2, y2, conf, cls)
            if cls in label_accept:
                new_rois.append((int(x1), int(y1), int(x2), int(y2)))
    return new_rois

def roi_selection_loop(source):
    """Mở video, chọn ROI bằng chuột hoặc tự động bằng YOLO."""
    global frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source", source)
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return

    clone = frame.copy()
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", draw_roi)

    print("Press 'd' to detect initial ROIs using YOLO. Press 'q' to quit.")

    while True:
        frame_display = clone.copy()

        # Vẽ ROI của YOLO
        for roi in roi_object:
            cv2.rectangle(frame_display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

        # Vẽ ROI do người dùng chọn
        for roi in rois_crew:
            cv2.rectangle(frame_display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

        cv2.imshow("Select ROI", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            detected_rois = yolo_detect_initial_rois(clone, model, [0, 1, 2, 3, 4])
            roi_object.extend(detected_rois)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
