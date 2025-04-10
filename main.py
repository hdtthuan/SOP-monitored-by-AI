import argparse
import subprocess


def main():
    # Initialize the argument parser to capture necessary flags.
    parser = argparse.ArgumentParser(
        description="Runner for main_module that forwards command-line flags."
    )

    # Define the flag for video source.
    parser.add_argument(
        "--source",
        type=str,
        default=r"data\wrong_video\WIN_20250326_09_38_37_Pro.mp4",
        help="Path to a video file or camera index (default: 0)",
    )

    # Define the flag for YOLO model weights.
    parser.add_argument(
        "--yolo_weight",
        type=str,
        default="./models/best_v4.pt",
        help="Path to the YOLO model weights file",
    )

    # Parse the incoming command-line arguments.
    args = parser.parse_args()

    # Continuous loop to execute main_module.py until the process meets the termination criteria.
    while True:
        # Assemble the subprocess command with the provided flags.
        command = [
            'python',
            'main_module.py',
            '--source', args.source,
            '--yolo_weight', args.yolo_weight
        ]

        # Execute main_module.py with the specified arguments.
        result = subprocess.run(command)

        # Break out of the loop if the return code from the module meets the exit condition.
        if result.returncode == 3:
            break


if __name__ == '__main__':
    main()
