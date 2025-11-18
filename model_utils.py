import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model_best.pth")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ResNet-18 with correct final FC matching the trained model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 3)
)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

class_names = ["HAPPY", "NEUTRAL", "SAD"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_emotion_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"error": "No face detected"}

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    label = class_names[int(np.argmax(probs))]
    return {"emotion": label}
