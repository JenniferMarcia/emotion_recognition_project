import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from app.config import CLASSES, IMG_SIZE

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, image: Image.Image, device):
    x = TRANSFORM(image.convert('RGB')).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


def print_prediction(name: str, probs: np.ndarray):
    order = np.argsort(-probs)
    print(f"\n{name}")
    print(f"  Prediction: {CLASSES[order[0]]} ({probs[order[0]] * 100:.1f}%)")
    for i in order:
        bar = '#' * int(probs[i] * 30)
        print(f"    {CLASSES[i]:<10} {probs[i] * 100:5.1f}% {bar}")
