"""
train_caries_classifier.py – Script pentru antrenarea modelului CNN ResNet-18

Acest script încarcă imaginile etichetate dindirectorul /dataset, aplică transformări de augmentare,
antrenează un model ResNet-18 pentru clasificarea imaginilor dentare în carie/non-carie și salvează cel mai bun model obținut în model_cnn.
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, f1_score

# Setări generale
train_dir = 'dataset/train'
val_dir = 'dataset/val'
batch_size = 16
num_epochs = 20
learning_rate = 0.001

# Selectare device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformări pentru augmentarea datelor și adaptarea la ResNet
transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Încărcarea dataset-urilor pentru antrenare și validare
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

print("Etichete:", train_dataset.class_to_idx)

class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1

class_weights = [1.0 / count for count in class_counts]
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoader cu sampler ponderat pentru antrenare
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Inițializare model ResNet18 pre-antrenat, cu ultimul strat modificat pentru 2 clase
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True  # Fine-tuning pe toate layerele
model.fc = nn.Linear(model.fc.in_features, 2)  # Clasificare binară
model = model.to(device)

# Funcție de pierdere și optimizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler pentru reducerea ratei de învățare după fiecare 5 epoci
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Antrenare model pe 20 de epoci
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoca {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # Evaluare pe setul de validare
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f" -> Acuratețe validare: {accuracy:.2f}%")
    print(f" -> F1-score (macro): {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Salvare model dacă este cel mai performant de până acum
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'model_cnn.pth')
        print(f"Model salvat (epoca {epoch+1}, {accuracy:.2f}%)")

    scheduler.step()
