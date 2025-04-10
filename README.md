# SOP-monitored-by-AI

## Overview
This repository houses an AI-powered solution for monitoring Standard Operating Procedure (SOP) adherence in real-time. The system leverages computer vision techniques to detect and track objects and hand movements to ensure proper procedural compliance.

## Key Features
- Real-time SOP compliance monitoring
- Object detection using YOLO models
- Hand tracking with MediaPipe
- Region of Interest (ROI) selection for focused analysis
- Timeline-based SOP action monitoring

## Project Structure
```
.
├── main.py                # Entry point; parses command-line args and starts application
├── config.py              # Configuration settings (model paths, default parameters)
├── requirements.txt       # List of dependencies
├── data/                  # Training and validation datasets
│   └── object-dataset/    # YOLO-formatted dataset with train, test, val subdirectories and data.yaml
├── models/
│   └── best.pt            # Pre-trained YOLO model file
├── modules/
│   ├── SOP_monitoring_v2.py  # Timeline action handling for SOP compliance
│   ├── roi_selector.py       # ROI selection and YOLO detection implementation
│   └── hand_detection_v2.py  # Hand detection using MediaPipe framework
├── research/              # Development workspace (optional)
└── utils/
    └── helper.py          # Utility functions (IoU calculation, visualization helpers)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hdtthuan/SOP-monitored-by-AI.git
   cd SOP-monitored-by-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train a new model on your custom dataset:
```bash
python train.py --data data/object-dataset/data.yaml --epochs 100 --batch 16
```

#### Training Options
```
--model_path    Path to the pretrained model weights (default: ./models/best_v4.pt)
--data          Path to the data configuration file (default: data.yaml)
--epochs        Number of training epochs (default: 1000)
--imgsz         Image size for training (default: 640)
--batch         Batch size for training (default: 8)
```

### Inference
To run the SOP monitoring on a video source:
```bash
python main.py --source data/Val_video.mp4 --yolo_weight models/best_v4.pt
```

#### Inference Options
```
--source        Path to input video file or webcam index (default: data\wrong_video\WIN_20250326_09_38_37_Pro.mp4)
--yolo_weight   Path to trained YOLO weights (default: ./models/best_v4.pt)
```

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics
- MediaPipe