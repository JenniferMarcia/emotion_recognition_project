"""
Test the ResNet18 FER2013 model locally.

python main.py --image photo.jpg
python main.py --dir some_folder
python main.py --test-dir ./Datasets/test
python main.py --webcam

Config via env vars (see .env.example): MODEL_PATH, DEVICE, CAMERA_INDEX, IMG_SIZE
"""
import argparse
from pathlib import Path

from app.config import DEVICE_OVERRIDE, MODEL_PATH
from app.evaluator import run_test_dir
from app.image_runner import run_dir, run_image
from app.model import load_model, resolve_device
from app.webcam_runner import run_webcam


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path)
    group.add_argument('--dir', type=Path)
    group.add_argument('--test-dir', type=Path)
    group.add_argument('--webcam', action='store_true')
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    device = resolve_device(DEVICE_OVERRIDE)
    print(f"Device: {device}")
    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, device)
    print("Model loaded.")

    if args.image:
        run_image(model, device, args.image)
    elif args.dir:
        run_dir(model, device, args.dir)
    elif args.test_dir:
        run_test_dir(model, device, args.test_dir)
    elif args.webcam:
        run_webcam(model, device)


if __name__ == '__main__':
    main()
