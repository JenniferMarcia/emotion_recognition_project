from pathlib import Path

import numpy as np
from PIL import Image

from app.config import CLASSES, IMG_EXTENSIONS
from app.predict import predict


def run_test_dir(model, device, test_dir: Path):
    samples = []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if cls_dir.name not in CLASSES:
            print(f"Warning: skipping unknown class folder '{cls_dir.name}'")
            continue
        cls_idx = CLASSES.index(cls_dir.name)
        samples += [(p, cls_idx) for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

    if not samples:
        print(f"No images found in {test_dir}")
        return

    print(f"Evaluating on {len(samples)} images...")
    preds = [int(np.argmax(predict(model, Image.open(p), device))) for p, _ in samples]
    labels = [label for _, label in samples]
    preds, labels = np.array(preds), np.array(labels)
    print(f"\nOverall accuracy: {(preds == labels).mean() * 100:.2f}%")

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n" + classification_report(labels, preds, target_names=CLASSES))
        print("Confusion matrix (rows=truth, cols=predicted):")
        print(confusion_matrix(labels, preds))
    except ImportError:
        print("(pip install scikit-learn for a detailed classification report)")
