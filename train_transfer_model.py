import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 0.0001
NUM_CLASSES = 3
DATA_DIR = "dataset"

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# DATA LOADERS
# -------------------------
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------------------------
# DEVICE
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# MODEL (Transfer Learning)
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze feature layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(device)

# -------------------------
# LOSS + OPTIMIZER
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# -------------------------
# TRAINING + VALIDATION
# -------------------------
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "emotion_model_best.pth")
        print(f"🔥 BEST MODEL UPDATED: {best_acc:.2f}%")

print("\nTraining Complete!")
print(f"⭐ Best Accuracy: {best_acc:.2f}%")
print("Saved as emotion_model_best.pth")
