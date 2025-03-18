import argparse
from roi_selection import roi_selection_loop
from hand_detection_v3 import hand_detection_loop


def main():
    parser = argparse.ArgumentParser(description="ROI selection and hand detection.")
    parser.add_argument("--source", type=str, default="/home/tuanphan/AI documents/FPT_AI_Semester/VIET DYNAMIC/SOP-monitored-by-AI/data/Val_video.mp4",
                        help="Path to video file or camera index (default: 0)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print("🔹 Bắt đầu chọn ROI...")
    roi_selection_loop(source)
    
    print("🔹 Bắt đầu nhận diện bàn tay...")
    hand_detection_loop(source)
 

if __name__ == "__main__":
    main()

