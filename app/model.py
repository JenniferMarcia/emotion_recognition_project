import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from app.config import CLASSES


def build_model(num_classes=len(CLASSES)):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_state_dict(model_path: Path):
    try:
        return torch.load(model_path, map_location='cpu')
    except (zipfile.BadZipFile, RuntimeError, KeyError):
        pass

    with zipfile.ZipFile(model_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith('.pth')]
        if not candidates:
            raise FileNotFoundError(f"No .pth file found inside {model_path}")
        with tempfile.TemporaryDirectory() as tmp:
            extracted = zf.extract(candidates[0], tmp)
            return torch.load(extracted, map_location='cpu')


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = build_model()
    model.load_state_dict(load_state_dict(model_path))
    model.to(device).eval()
    return model


def resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
