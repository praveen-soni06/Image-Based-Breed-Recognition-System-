import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
import torchvision.models as models
import optuna
# Set random seeds for reproducibility
torch.manual_seed(42)
# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data_path = "D:\SH\dataset"
print(os.listdir(data_path))
images = []
labels = []

for breed_folder in os.listdir(data_path):
    breed_path = os.path.join(data_path, breed_folder)
    if os.path.isdir(breed_path):
        for image_name in os.listdir(breed_path):
            img_path = os.path.join(breed_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))  # Bigger size for ResNet
                images.append(img)
                labels.append(breed_folder)

X = np.array(images, dtype="uint8")  # keep raw images
y = np.array(labels)
# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# CUSTOM DATASET
class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img = self.features[idx].astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
# Evaluation
def evaluate(model, loader, device):
    model.eval()
    top1_correct, top3_correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top3 = outputs.topk(3, dim=1)
            total += labels.size(0)
            top1_correct += (pred_top1.squeeze() == labels).sum().item()
            top3_correct += sum([labels[i] in pred_top3[i] for i in range(labels.size(0))])
    return 100 * top1_correct / total, 100 * top3_correct / total
# Optuna objective
def objective(trial):
    # HYPERPARAMETERS TO TUNE
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    freeze_layers = trial.suggest_categorical("freeze_layers", ["layer2+", "layer3+", "layer4+"])
    
    # AUGMENTATION HYPERPARAMS
    random_flip = trial.suggest_float("random_flip", 0.0, 0.7)
    rotation = trial.suggest_int("rotation", 0, 25)
    color_jitter = trial.suggest_float("color_jitter", 0.0, 0.2)
    random_erasing_prob = trial.suggest_float("random_erasing_prob", 0.0, 0.3)

    # TRANSFORMS
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=random_flip),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                               saturation=color_jitter, hue=color_jitter/4),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    test_dataset = CustomDataset(X_test, y_test, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    # MODEL
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Freeze layers based on trial
    for name, param in model.named_parameters():
        param.requires_grad = True
        if freeze_layers == "layer2+" and "layer1" in name:
            param.requires_grad = False
        elif freeze_layers == "layer3+" and ("layer1" in name or "layer2" in name):
            param.requires_grad = False
        elif freeze_layers == "layer4+" and ("layer1" in name or "layer2" in name or "layer3" in name):
            param.requires_grad = False

    model = model.to(device)

    # LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # TRAINING LOOP
    epochs = 10  # keep small for faster Optuna tuning
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels_batch in train_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        
        # Step scheduler using training loss
        scheduler.step(avg_loss)
    
    # EVALUATE
    top1, top3 = evaluate(model, test_loader, device)
    
    # Optuna tries to maximize top-1 accuracy
    return top1

# RUN OPTUNA STUDY
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # you can increase n_trials later

print("Best trial:")
trial = study.best_trial
print("  Top-1 Accuracy:", trial.value)
print("  Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")