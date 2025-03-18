import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

# Global parameters
CROP_WIDTH = 350
CROP_HEIGHT = 350
IMG_HEIGHT = CROP_HEIGHT
IMG_WIDTH = CROP_WIDTH
FLOW_WINDOW = 3  # Number of frames for averaging optical flow
DROPOUT = 0.5

# Threshold for detecting a significant change in crop center (in pixels)
CENTER_CHANGE_THRESHOLD = 50

# ---------------------------
# Model Definition
# ---------------------------
class ThreeStreamModel(torch.nn.Module):
    def __init__(self, img_height, img_width, num_classes):
        super(ThreeStreamModel, self).__init__()

        def create_branch():
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 7 * 7, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(DROPOUT)
            )

        self.spatial_branch = create_branch()
        self.temporal_branch = create_branch()
        self.pose_branch = create_branch()
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(128 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, spatial, temporal, pose):
        spatial_feat = self.spatial_branch(spatial)
        temporal_feat = self.temporal_branch(temporal)
        pose_feat = self.pose_branch(pose)
        fused = torch.cat([spatial_feat, temporal_feat, pose_feat], dim=1)
        out = self.fusion(fused)
        return out


# ---------------------------
# Helper Function: Convert Hand Landmarks to Skeleton Image
# ---------------------------
def convert_hands_to_image(hand_landmarks_list, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if hand_landmarks_list is None:
        return img
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(img, (x, y), radius=3, color=(255, 255, 255), thickness=-1)
    return img


# ---------------------------
# Global State Variables
# ---------------------------
last_center_x = None
last_center_y = None
prev_gray_crop = None
flow_list = []

# Global device, label map, and model initialization.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "screwdriver_action", 1: "unknown"}
num_classes = len(label_map)
model = ThreeStreamModel(IMG_HEIGHT, IMG_WIDTH, num_classes).to(device)
model.load_state_dict(torch.load("models/three_stream_model_screw_driver.pth", map_location=device))
model.eval()


# ---------------------------
# Single-Function Prediction (Input: one frame)
# ---------------------------
def process_frame(frame, result_hands, CONF_THRESHOLD=0.7):
    """
    Accepts a single BGR frame and a pre-computed MediaPipe result (result_hands).
    Processes spatial, temporal, and pose streams, runs prediction, overlays prediction,
    and returns the output frame along with predicted label and confidence.
    """
    global last_center_x, last_center_y, prev_gray_crop, flow_list

    # Flip frame for selfie-view and make a copy.
    # frame = cv2.flip(frame, 1)
    frame_nodraw = frame.copy()

    # Use provided result_hands (assumed to be computed outside this function).
    results = result_hands

    # Determine crop coordinates using hand landmarks.
    x_min, y_min, x_max, y_max = None, None, None, None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hx_min = int(min(xs) * frame.shape[1])
            hy_min = int(min(ys) * frame.shape[0])
            hx_max = int(max(xs) * frame.shape[1])
            hy_max = int(max(ys) * frame.shape[0])
            if x_min is None or hx_min < x_min:
                x_min = hx_min
            if y_min is None or hy_min < y_min:
                y_min = hy_min
            if x_max is None or hx_max > x_max:
                x_max = hx_max
            if y_max is None or hy_max > y_max:
                y_max = hy_max

    # Determine center based on landmarks or use previous center.
    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        # If there's a large jump in center, reset optical flow history.
        if last_center_x is not None:
            if abs(center_x - last_center_x) > CENTER_CHANGE_THRESHOLD or abs(center_y - last_center_y) > CENTER_CHANGE_THRESHOLD:
                flow_list = []
                prev_gray_crop = None
        last_center_x, last_center_y = center_x, center_y
    else:
        center_x = last_center_x if last_center_x is not None else frame.shape[1] // 2
        center_y = last_center_y if last_center_y is not None else frame.shape[0] // 2

    # Calculate crop coordinates.
    x_min_crop = max(center_x - CROP_WIDTH // 2, 0)
    y_min_crop = max(center_y - CROP_HEIGHT // 2, 0)
    x_max_crop = min(center_x + CROP_WIDTH // 2, frame.shape[1])
    y_max_crop = min(center_y + CROP_HEIGHT // 2, frame.shape[0])

    # Spatial stream: crop and resize.
    spatial_crop = frame_nodraw[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    spatial_crop = cv2.resize(spatial_crop, (IMG_WIDTH, IMG_HEIGHT))

    # Temporal stream: compute optical flow.
    current_gray = cv2.cvtColor(spatial_crop, cv2.COLOR_BGR2GRAY)
    if prev_gray_crop is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray_crop, current_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        flow_list.append(flow)
        # Keep only the last FLOW_WINDOW flows.
        flow_list = flow_list[-FLOW_WINDOW:]
    prev_gray_crop = current_gray.copy()

    if len(flow_list) < FLOW_WINDOW:
        temporal_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    else:
        avg_flow = np.mean(np.array(flow_list), axis=0)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])
        hsv = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        temporal_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Pose stream: generate a skeleton image.
    pose_img = convert_hands_to_image(results.multi_hand_landmarks, IMG_WIDTH, IMG_HEIGHT)

    # Preprocess images: scale pixel values and convert to torch tensors.
    spatial_input = torch.from_numpy(spatial_crop.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    temporal_input = torch.from_numpy(temporal_img.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pose_input = torch.from_numpy(pose_img.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run model prediction.
    with torch.no_grad():
        outputs = model(spatial_input, temporal_input, pose_input)
        probabilities = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()

    pred_label = None
    if confidence >= CONF_THRESHOLD and pred_class == 0:
        # pred_label = label_map.get(pred_class, "Unknown")
        # cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.rectangle(frame, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop), (255, 0, 0), 2)
        pred_label = label_map.get(pred_class, "Unknown")

    return frame, pred_label, confidence
