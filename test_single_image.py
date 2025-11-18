import torch
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 128
NUM_CLASSES = 3
LABELS = ["happy", "neutral", "sad"]  # ORDER MUST MATCH YOUR FOLDER ORDER

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("emotion_model_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Load test image
img_path = "test.jpg"  # change this
img = Image.open(img_path)

img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

print("Predicted Emotion:", LABELS[pred.item()])
