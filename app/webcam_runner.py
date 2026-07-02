import numpy as np
from PIL import Image

from app.config import CAMERA_INDEX, CLASSES
from app.predict import predict


def run_webcam(model, device):
    try:
        import cv2
    except ImportError:
        raise SystemExit("--webcam requires opencv-python: pip install opencv-python")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam (index {CAMERA_INDEX})")

    print("Webcam open. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
                probs = predict(model, Image.fromarray(face_rgb), device)
                label = f"{CLASSES[int(np.argmax(probs))]} {probs.max() * 100:.0f}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Emotion Recognition - ResNet18 FER2013', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
