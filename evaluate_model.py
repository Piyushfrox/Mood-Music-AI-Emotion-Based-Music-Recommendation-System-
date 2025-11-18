import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------
#  CONFIG
# -------------------------
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_CLASSES = 3
DATA_DIR = "dataset"

# -------------------------
#  TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------------
#  LOAD VAL DATA
# -------------------------
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------------------------
#  LOAD MODEL
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------
#  ACCURACY TEST
# -------------------------
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = correct / total * 100
print(f"\n🎯 Validation Accuracy: {accuracy:.2f}%\n")
