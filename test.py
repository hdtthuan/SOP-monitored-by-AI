import cv2
from ultralytics import YOLO
import numpy as np

# Khởi tạo model YOLOv8 (chỉnh path thành model của bạn)
model_path = "./models/best1.pt"  # Hoặc yolov8s.pt, yolov8m.pt,...
model = YOLO(model_path)

# Mở video (0: webcam, hoặc thay bằng đường dẫn file)
video_path = r"C:\Users\GMT\SOP-monitored-by-AI\data\Val_video.mp4"
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu không mở được video
if not cap.isOpened():
    print("Không thể mở video")
    exit()

def adjust_gamma(image, gamma=0.5):  # Giảm sáng (gamma < 1)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame, kết thúc!")
        break


    frame = cv2.flip(frame, 1)    
    frame = adjust_gamma(frame, gamma=0.5)

    # Phát hiện đối tượng
    results = model(frame, conf=0.2, iou=0.9)

    # Duyệt qua từng kết quả
    for r in results:
        for box in r.boxes.data.tolist():  # [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls = box
            label = model.names[int(cls)]
            
            # Vẽ khung chữ nhật
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # Hiển thị kết quả
    cv2.imshow("YOLO Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
