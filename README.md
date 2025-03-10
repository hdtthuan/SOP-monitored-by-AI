# SOP-monitored-by-AI

Description.

## Folder Structure

```
.
├── main.py                # Entry point; parses command-line args and starts the application
├── config.py              # Configuration settings (e.g., default source, model paths)
├── requirements.txt       # List of dependencies (opencv-python, mediapipe, ultralytics, etc.)
├── data                   # data for train or validation.
├── models/
│   └── best.pt            # Pre-trained YOLO model file
├── modules/
│   ├── camera_handler.py  # Handles video capture (opens camera or video file)
│   ├── roi_selector.py    # ROI selection and YOLO detection routines
│   └── hand_detector.py   # Hand detection routines using MediaPipe
├── research/              # workspace of each dev
└── utils/
    └── helper.py          # Utility functions (e.g., IoU calculation, drawing helpers)
```