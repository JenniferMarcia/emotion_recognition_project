import os
from pathlib import Path

MODEL_PATH = Path(os.getenv('MODEL_PATH', 'resnet18_fer2013.pth.zip'))
DEVICE_OVERRIDE = os.getenv('DEVICE')
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
IMG_SIZE = int(os.getenv('IMG_SIZE', '224'))

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
