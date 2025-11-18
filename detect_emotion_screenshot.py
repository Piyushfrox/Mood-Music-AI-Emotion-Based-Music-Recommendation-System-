import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import time
from play_music import play_music

# ----------------------
# Load Model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

class_names = ["happy", "neutral", "sad"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ----------------------
# Haar Cascade
# ----------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def predict_emotion(face_img):
    img = Image.fromarray(face_img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    idx = np.argmax(probs)
    return class_names[idx], probs[idx]

# ----------------------
# Webcam
# ----------------------
cap = cv2.VideoCapture(0)

last_prediction = None
last_time = 0
cooldown = 10   # <-- 10 seconds delay

print("🎥 Webcam started... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Only predict every cooldown seconds
        if time.time() - last_time > cooldown:
            emotion, conf = predict_emotion(face)
            last_prediction = f"{emotion} ({conf*100:.1f}%)"
            print("Detected:", last_prediction)

            # Play music only when emotion changes
            play_music(emotion)

            last_time = time.time()

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write emotion on screen
        if last_prediction:
            cv2.putText(frame, last_prediction, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Mood Music - Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
