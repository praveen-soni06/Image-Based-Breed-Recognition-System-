# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pickle
from pathlib import Path

# -------- CONFIG --------
MODEL_PATH = Path("D:/SH/SIH2025/ML Model/best_breed_model1.pth")
LE_PATH = Path("D:/SH/SIH2025/ML Model/label_encoder.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD MODEL --------
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------- FASTAPI APP --------
app = FastAPI(title="Breed Recognition API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- PREDICTION ENDPOINT --------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            breed = le.inverse_transform(top_idx.cpu().numpy())[0]

        return {"breed": breed, "confidence": float(top_prob.cpu().item())}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------- HEALTH CHECK --------
@app.get("/")
async def root():
    return {"message": "Breed Recognition API is running ✅"}
