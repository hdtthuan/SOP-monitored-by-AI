import cv2
from ultralytics import YOLO

# ===== Biến Toàn Cục =====

rois_crew = []  # Lưu ROIs được chọn thủ công
roi_object = {}  # object_name: roi
selecting = False  # Đánh dấu trạng thái đang chọn ROI
start_point = None  # Điểm bắt đầu khi vẽ ROI
frame = None  # Frame video hiện tại

# Khởi tạo YOLO model
from ultralytics import YOLO

class ROISelector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # ✅ Use dynamic model path

    def select_roi(self, frame):
        results = self.model(frame)
        return results



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
            cv2.rectangle(
                frame_copy, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2
            )
            cv2.putText(
                frame_copy,
                f"Crew {i + 1}",
                (roi[0], roi[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        # Draw the ROI being currently selected
        cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:  # Finalize the ROI selection
        rois_crew.append((start_point[0], start_point[1], x, y))
        selecting = False


# def yolo_detect_initial_rois(frame, model, label_accept=[]) -> dict:
#     """
#     Uses YOLO to detect objects in the current frame and returns their bounding boxes as ROIs.
#     """
#     results = model(frame, conf=0.1)
#     detected_rois = {}
#     # YOLOv8 returns a list of Results objects (one per image)
#     for r in results:
#         for box in r.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box
#             print(x1, y1, x2, y2, conf, cls)
#             if cls in label_accept:
#                 detected_rois[model.names[int(cls)]] = ((int(x1), int(y1), int(x2), int(y2)))
#     return detected_rois
def yolo_detect_initial_rois(frame, model, label_accept=[]):
    """
    Uses YOLO to detect objects in the current frame and returns only the highest confidence ROI per class.
    """
    results = model(frame, conf=0.1)
    best_rois = {}  # Lưu ROI có độ tin cậy cao nhất của mỗi class

    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)  # Chuyển class index thành integer

            if cls in label_accept:
                # Nếu chưa có ROI cho class này, hoặc confidence cao hơn thì cập nhật
                if cls not in best_rois or conf > best_rois[cls][1]:
                    best_rois[cls] = ((int(x1), int(y1), int(x2), int(y2)), conf)

    # Chuyển đổi dictionary để chỉ lấy tọa độ ROI tốt nhất
    detected_rois = {model.names[cls]: roi[0] for cls, roi in best_rois.items()}
    return detected_rois


def roi_selection_loop(source, roi_selector):
    """Mở video, chọn ROI bằng chuột hoặc tự động bằng YOLO."""
    global frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source", source)
        exit(3)

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
        for object_name, roi in roi_object.items():
            cv2.rectangle(
                frame_display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2
            )
            cv2.putText(
                frame,
                f"{object_name}",
                (roi[0], roi[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Vẽ ROI do người dùng chọn
        for i, roi in enumerate(rois_crew):
            cv2.rectangle(
                frame_display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Crew {i + 1}",
                (roi[0], roi[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Select ROI", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            detected_rois = yolo_detect_initial_rois(clone, roi_selector.model, [0, 1, 2, 3, 4])
            roi_object.update(detected_rois)
            print(roi_object)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
