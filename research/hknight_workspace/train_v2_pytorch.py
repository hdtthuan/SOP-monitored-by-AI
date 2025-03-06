import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ----- Parameters and Global Settings -----
CROP_WIDTH = 350
CROP_HEIGHT = 350

# Folder structure:
#   spatial_dataset/<action_label>/<video_name>/frame_xxxx.jpg
#   temporal_dataset/<action_label>/<video_name>/frame_xxxx.jpg
#   pose_dataset/<action_label>/<video_name>/frame_xxxx.jpg
DATASET_DIR = "data/labels_videos_actions/output_convert"
SPATIAL_OUTPUT_DIR = "data/labels_videos_actions/spatial_dataset"
TEMPORAL_OUTPUT_DIR = "data/labels_videos_actions/temporal_dataset"
POSE_OUTPUT_DIR = "data/labels_videos_actions/pose_dataset"

# Training parameters
IMG_HEIGHT = CROP_HEIGHT
IMG_WIDTH = CROP_WIDTH
BATCH_SIZE = 4  # Adjust as needed
EPOCHS = 30
MODEL_SAVE_PATH = "three_stream_model.pth"
FLOW_WINDOW = 10  # Number of consecutive flows to average for temporal stream
DROPOUT = 0.5

# ----- Helper Function: Convert Hand Landmarks to Skeleton Image (for Pose Stream) -----
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


# ----- Video Processing (Extract Three Streams: Spatial, Temporal & Pose) -----
def process_video(video_path, spatial_output_folder, temporal_output_folder, pose_output_folder,
                  flow_window=FLOW_WINDOW):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    desired_fps = 25
    frame_interval = max(1, int(round(actual_fps / desired_fps)))
    print(f"Actual FPS: {actual_fps}, using frame interval: {frame_interval} to achieve {desired_fps} FPS")

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.02,
                        min_tracking_confidence=0.1,
                        max_num_hands=2) as hands:
        last_center_x = None
        last_center_y = None
        frame_index = 0
        saved_frame_count = 0
        prev_gray_crop = None
        flow_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval != 0:
                frame_index += 1
                continue
            frame_index += 1

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            x_min, y_min, x_max, y_max = None, None, None, None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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

            if results.multi_hand_landmarks:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                last_center_x, last_center_y = center_x, center_y
            else:
                center_x = last_center_x if last_center_x is not None else frame.shape[1] // 2
                center_y = last_center_y if last_center_y is not None else frame.shape[0] // 2

            x_min_crop = int(max(center_x - CROP_WIDTH / 2, 0))
            y_min_crop = int(max(center_y - CROP_HEIGHT / 2, 0))
            x_max_crop = int(min(center_x + CROP_WIDTH / 2, frame.shape[1]))
            y_max_crop = int(min(center_y + CROP_HEIGHT / 2, frame.shape[0]))
            spatial_crop = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            spatial_crop = cv2.resize(spatial_crop, (CROP_WIDTH, CROP_HEIGHT))

            current_gray = cv2.cvtColor(spatial_crop, cv2.COLOR_BGR2GRAY)
            if prev_gray_crop is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray_crop, current_gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                flow_list.append(flow)
            prev_gray_crop = current_gray.copy()

            if len(flow_list) < flow_window:
                flow_img = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
            else:
                avg_flow = np.mean(np.array(flow_list[-flow_window:]), axis=0)
                mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])
                hsv = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            pose_img = convert_hands_to_image(results.multi_hand_landmarks, CROP_WIDTH, CROP_HEIGHT)

            spatial_path = os.path.join(spatial_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            temporal_path = os.path.join(temporal_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            pose_path = os.path.join(pose_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(spatial_path, spatial_crop)
            cv2.imwrite(temporal_path, flow_img)
            cv2.imwrite(pose_path, pose_img)
            saved_frame_count += 1

    cap.release()


def process_dataset(dataset_dir, spatial_output_base, temporal_output_base, pose_output_base):
    for out_dir in [spatial_output_base, temporal_output_base, pose_output_base]:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if os.path.isdir(label_path):
            spatial_label_folder = os.path.join(spatial_output_base, label)
            temporal_label_folder = os.path.join(temporal_output_base, label)
            pose_label_folder = os.path.join(pose_output_base, label)
            for folder in [spatial_label_folder, temporal_label_folder, pose_label_folder]:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            for video_file in os.listdir(label_path):
                if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_path = os.path.join(label_path, video_file)
                    video_name = os.path.splitext(video_file)[0]
                    spatial_video_folder = os.path.join(spatial_label_folder, video_name)
                    temporal_video_folder = os.path.join(temporal_label_folder, video_name)
                    pose_video_folder = os.path.join(pose_label_folder, video_name)
                    for folder in [spatial_video_folder, temporal_video_folder, pose_video_folder]:
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                    print(f"Processing video: {video_path}")
                    process_video(video_path, spatial_video_folder, temporal_video_folder, pose_video_folder)


# ----- Three-Stream Dataset (PyTorch DataLoader) -----
class ThreeStreamDataset(Dataset):
    def __init__(self, spatial_dir, temporal_dir, pose_dir, target_size, transform=None):
        self.spatial_dir = spatial_dir
        self.temporal_dir = temporal_dir
        self.pose_dir = pose_dir
        self.target_size = target_size  # (width, height) for PIL
        self.transform = transform
        self.samples = []

        for label in sorted(os.listdir(spatial_dir)):
            spatial_label_path = os.path.join(spatial_dir, label)
            temporal_label_path = os.path.join(temporal_dir, label)
            pose_label_path = os.path.join(pose_dir, label)
            if os.path.isdir(spatial_label_path) and os.path.isdir(temporal_label_path) and os.path.isdir(
                    pose_label_path):
                for video in os.listdir(spatial_label_path):
                    spatial_video_path = os.path.join(spatial_label_path, video)
                    temporal_video_path = os.path.join(temporal_label_path, video)
                    pose_video_path = os.path.join(pose_label_path, video)
                    if os.path.isdir(spatial_video_path) and os.path.isdir(temporal_video_path) and os.path.isdir(
                            pose_video_path):
                        spatial_frames = sorted([os.path.join(spatial_video_path, f)
                                                 for f in os.listdir(spatial_video_path) if f.endswith(".jpg")])
                        temporal_frames = sorted([os.path.join(temporal_video_path, f)
                                                  for f in os.listdir(temporal_video_path) if f.endswith(".jpg")])
                        pose_frames = sorted([os.path.join(pose_video_path, f)
                                              for f in os.listdir(pose_video_path) if f.endswith(".jpg")])
                        for s_path, t_path, p_path in zip(spatial_frames, temporal_frames, pose_frames):
                            self.samples.append((s_path, t_path, p_path, label))
        labels_set = sorted(list(set([s[3] for s in self.samples])))
        self.class_indices = {label: idx for idx, label in enumerate(labels_set)}
        self.samples = [(s_path, t_path, p_path, self.class_indices[label]) for s_path, t_path, p_path, label in
                        self.samples]
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_path, t_path, p_path, label = self.samples[idx]
        s_img = Image.open(s_path).convert('RGB').resize(self.target_size)
        t_img = Image.open(t_path).convert('RGB').resize(self.target_size)
        p_img = Image.open(p_path).convert('RGB').resize(self.target_size)

        if self.transform:
            s_img = self.transform(s_img)
            t_img = self.transform(t_img)
            p_img = self.transform(p_img)
        else:
            s_img = transforms.ToTensor()(s_img)
            t_img = transforms.ToTensor()(t_img)
            p_img = transforms.ToTensor()(p_img)

        return (s_img, t_img, p_img), label


# ----- Three-Stream Model Architecture (PyTorch) -----
class ThreeStreamModel(nn.Module):
    def __init__(self, img_height, img_width, num_classes):
        super(ThreeStreamModel, self).__init__()

        # Each branch follows: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> AdaptiveAvgPool -> Flatten -> FC -> ReLU -> Dropout.
        # To avoid extremely high-dimensional flattened vectors, we use AdaptiveAvgPool2d.
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


# ----- Model Training (Using the Three-Stream Network in PyTorch) -----
def train_three_stream_model(spatial_dataset_dir, temporal_dataset_dir, pose_dataset_dir, batch_size, epochs,
                             target_size, device):
    # Define data augmentation transforms (applied during dataset loading)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])
    # Note: target_size is (width, height) for PIL
    dataset = ThreeStreamDataset(spatial_dataset_dir, temporal_dataset_dir, pose_dataset_dir, target_size,
                                 transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.class_indices)
    model = ThreeStreamModel(IMG_HEIGHT, IMG_WIDTH, num_classes)
    model.to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing weights from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Compute class weights
    labels = [sample[-1] for sample in dataset.samples]
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters())
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    # For demonstration, we use the same dataloader for training and validation.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for (spatial, temporal, pose), labels in dataloader:
            spatial = spatial.to(device)
            temporal = temporal.to(device)
            pose = pose.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(spatial, temporal, pose)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * spatial.size(0)

        epoch_loss = running_loss / len(dataset)

        # Validation pass (using the same dataloader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (spatial, temporal, pose), labels in dataloader:
                spatial = spatial.to(device)
                temporal = temporal.to(device)
                pose = pose.to(device)
                labels = labels.to(device)
                outputs = model(spatial, temporal, pose)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * spatial.size(0)
        val_loss /= len(dataset)
        # Step the scheduler at the end of each epoch
        # scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model improved. Saving weights to {MODEL_SAVE_PATH}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return model


# ----- Main Pipeline -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Extracting frames from videos for spatial, temporal (optical flow), and pose streams...")
    # Uncomment the following line to run the frame extraction
    # process_dataset(DATASET_DIR, SPATIAL_OUTPUT_DIR, TEMPORAL_OUTPUT_DIR, POSE_OUTPUT_DIR)
    print("Frame extraction complete.")

    print("Starting three-stream model training...")
    # Note: target_size for PIL is given as (width, height)
    model = train_three_stream_model(SPATIAL_OUTPUT_DIR, TEMPORAL_OUTPUT_DIR, POSE_OUTPUT_DIR,
                                     BATCH_SIZE, EPOCHS, (IMG_WIDTH, IMG_HEIGHT), device)
    print(f"Model saved to {MODEL_SAVE_PATH}")
