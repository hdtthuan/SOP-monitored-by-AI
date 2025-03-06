import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Parameters and Global Settings -----
CROP_WIDTH = 350
CROP_HEIGHT = 350
IMG_HEIGHT = CROP_HEIGHT
IMG_WIDTH = CROP_WIDTH
MODEL_SAVE_PATH = "three_stream_model_bk0.3.pth"  # Path to your trained PyTorch three-stream model
CONF_THRESHOLD = 0.95  # Only overlay prediction if confidence is above this
FLOW_WINDOW = 3  # Number of flows to average for multi-frame optical flow
DROPOUT = 0.3

# Initialize MediaPipe drawing utilities (for visualizing hand landmarks)
mp_drawing = mp.solutions.drawing_utils


# ----- Helper Function: Convert Hand Landmarks to Skeleton Image (Pose Stream) -----
def convert_hands_to_image(hand_landmarks_list, width, height):
    """
    Converts MediaPipe hand landmarks into a skeleton-like image by drawing white circles
    on a black background. If no landmarks are provided, returns a black image.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if hand_landmarks_list is None:
        return img
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(img, (x, y), radius=3, color=(255, 255, 255), thickness=-1)
    return img


# ----- Three-Stream Model Architecture (PyTorch) -----
class ThreeStreamModel(nn.Module):
    def __init__(self, img_height, img_width, num_classes):
        super(ThreeStreamModel, self).__init__()

        def create_branch():
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            )

        self.spatial_branch = create_branch()
        self.temporal_branch = create_branch()
        self.pose_branch = create_branch()
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, spatial, temporal, pose):
        spatial_feat = self.spatial_branch(spatial)
        temporal_feat = self.temporal_branch(temporal)
        pose_feat = self.pose_branch(pose)
        fused = torch.cat([spatial_feat, temporal_feat, pose_feat], dim=1)
        out = self.fusion(fused)
        return out


# ----- Prediction Function for Camera Feed with Rotation -----
def predict_camera(model_path, label_map, device, camera_index=0, rotation_angle=0):
    """
    Capture frames from the camera, process them to generate three streams:
      - Spatial: Cropped RGB image from the original (unannotated) frame.
      - Temporal: Multi-frame optical flow computed from consecutive cropped grayscale frames.
      - Pose: Skeleton image generated from MediaPipe Hands landmarks.
    The three inputs are fed into the three-stream model for action prediction.

    Optionally rotates each captured frame by the specified rotation_angle (in degrees).
    """
    num_classes = len(label_map)
    # Initialize the model and load weights.
    model = ThreeStreamModel(IMG_HEIGHT, IMG_WIDTH, num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
        return
    model.eval()

    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.3,
                        min_tracking_confidence=0.3,
                        max_num_hands=2) as hands:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error opening camera with index {camera_index}")
            return

        # Try to determine camera FPS; if unavailable, assume 30.
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps == 0:
            actual_fps = 30
        desired_fps = 25
        frame_interval = max(1, int(round(actual_fps / desired_fps)))
        print(
            f"Camera actual FPS: {actual_fps}, sampling every {frame_interval} frame(s) to achieve ~{desired_fps} FPS")

        last_center_x = None
        last_center_y = None
        frame_index = 0

        # For multi-frame optical flow accumulation.
        prev_gray_crop = None
        flow_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if specified.
            if rotation_angle != 0:
                (h, w) = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

            # Process only every n-th frame.
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue
            frame_index += 1

            # Flip frame (selfie-view) and create a copy without annotations.
            frame = cv2.flip(frame, 1)
            frame_nodraw = frame.copy()  # This copy will be used for spatial cropping and optical flow.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Hands.
            results = hands.process(rgb_frame)
            x_min, y_min, x_max, y_max = None, None, None, None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks for visualization on the annotated frame.
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_x_coords = [lm.x for lm in hand_landmarks.landmark]
                    hand_y_coords = [lm.y for lm in hand_landmarks.landmark]
                    hand_x_min = int(min(hand_x_coords) * frame.shape[1])
                    hand_y_min = int(min(hand_y_coords) * frame.shape[0])
                    hand_x_max = int(max(hand_x_coords) * frame.shape[1])
                    hand_y_max = int(max(hand_y_coords) * frame.shape[0])
                    if x_min is None or hand_x_min < x_min:
                        x_min = hand_x_min
                    if y_min is None or hand_y_min < y_min:
                        y_min = hand_y_min
                    if x_max is None or hand_x_max > x_max:
                        x_max = hand_x_max
                    if y_max is None or hand_y_max > y_max:
                        y_max = hand_y_max

            # Determine crop center based on detected hands or fallback to previous/center.
            if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                last_center_x, last_center_y = center_x, center_y
            else:
                center_x = last_center_x if last_center_x is not None else frame.shape[1] // 2
                center_y = last_center_y if last_center_y is not None else frame.shape[0] // 2

            # Compute crop coordinates.
            x_min_crop = int(max(center_x - CROP_WIDTH / 2, 0))
            y_min_crop = int(max(center_y - CROP_HEIGHT / 2, 0))
            x_max_crop = int(min(center_x + CROP_WIDTH / 2, frame.shape[1]))
            y_max_crop = int(min(center_y + CROP_HEIGHT / 2, frame.shape[0]))

            # Use the non-annotated copy for spatial cropping.
            spatial_crop = frame_nodraw[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            spatial_crop = cv2.resize(spatial_crop, (IMG_WIDTH, IMG_HEIGHT))

            # ----- Temporal Stream: Multi-frame Optical Flow -----
            current_gray = cv2.cvtColor(spatial_crop, cv2.COLOR_BGR2GRAY)
            if prev_gray_crop is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray_crop, current_gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                flow_list.append(flow)
            prev_gray_crop = current_gray.copy()

            # Use a black image if not enough flows have been accumulated.
            if len(flow_list) < FLOW_WINDOW:
                temporal_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            else:
                avg_flow = np.mean(np.array(flow_list[-FLOW_WINDOW:]), axis=0)
                mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])
                hsv = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                hsv[..., 0] = ang * 180 / np.pi / 2  # Flow direction as hue.
                hsv[..., 1] = 255  # Full saturation.
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude.
                temporal_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # ----- Pose Stream: Hand Skeleton (from the annotated frame) -----
            pose_img = convert_hands_to_image(results.multi_hand_landmarks, IMG_WIDTH, IMG_HEIGHT)

            # Preprocess the images: normalize pixel values to [0, 1].
            spatial_img = spatial_crop.astype("float32") / 255.0
            temporal_img_norm = temporal_img.astype("float32") / 255.0
            pose_img_norm = pose_img.astype("float32") / 255.0

            # Convert NumPy arrays to torch tensors with channels-first format.
            spatial_input = torch.from_numpy(spatial_img).permute(2, 0, 1).unsqueeze(0).to(device)
            temporal_input = torch.from_numpy(temporal_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            pose_input = torch.from_numpy(pose_img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

            # Predict using the three-stream model.
            with torch.no_grad():
                outputs = model(spatial_input, temporal_input, pose_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            # Overlay prediction if confidence exceeds threshold.
            if confidence >= CONF_THRESHOLD:
                predicted_label = label_map.get(predicted_class, "Unknown")
                text = f"{predicted_label} ({confidence:.2f})"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop),
                              (255, 0, 0), 2)

            # Display the annotated frame and the generated streams.
            cv2.imshow("Three-Stream Prediction", frame)
            cv2.imshow("Temporal Optical Flow", temporal_img)
            cv2.imshow("Pose Skeleton", pose_img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict action from live camera feed using a three-stream model (PyTorch)."
    )
    parser.add_argument("--camera", type=int, default=2,
                        help="Camera index to use (default is 2).")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH,
                        help="Path to the trained three-stream model (.pth file).")
    parser.add_argument("--labels", type=str, default="action1,action2,action3",
                        help="Comma-separated list of labels corresponding to model outputs.")
    parser.add_argument("--rotation", type=float, default=0,
                        help="Rotation angle in degrees to apply to the camera frames (default: 0).")
    args = parser.parse_args()

    # Create a label map dictionary from the comma-separated list.
    label_list = [label.strip() for label in args.labels.split(",")]
    label_map = {i: label for i, label in enumerate(label_list)}

    # Set device to GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_camera(args.model, label_map, device, camera_index=args.camera, rotation_angle=args.rotation)
