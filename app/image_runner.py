from pathlib import Path

from PIL import Image

from app.config import IMG_EXTENSIONS
from app.predict import predict, print_prediction


def run_image(model, device, image_path: Path):
    print_prediction(image_path.name, predict(model, Image.open(image_path), device))


def run_dir(model, device, dir_path: Path):
    images = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    if not images:
        print(f"No images found in {dir_path}")
        return
    for img_path in images:
        try:
            run_image(model, device, img_path)
        except Exception as e:
            print(f"\n{img_path.name}\n  Error: {e}")
