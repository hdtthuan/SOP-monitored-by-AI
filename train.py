import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model with custom hyperparameters"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/best_v4.pt",
        help="Path to the pretrained model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to the data configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the pretrained model
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)

    # Start training with the provided hyperparameters
    print("Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    print("Training complete.")


if __name__ == '__main__':
    main()
