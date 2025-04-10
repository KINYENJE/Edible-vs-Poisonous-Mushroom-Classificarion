from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os

app = FastAPI()

model_path = "models/mushroom_img_model.h5"
google_drive_url = "https://drive.google.com/uc?id=15tsi1T_RllQ08yplYbm1dfYm6TAoeuKD"

# Allow CORS for all origins (adjust as needed for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if the model file exists, and download it if not
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Downloading model from Google Drive...")
    response = requests.get(google_drive_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# Load the model
model = load_model(model_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the Mushroom Classification API!"}



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    preds = model.predict(image)
    result = "Poisonous" if preds[0] > 0.6 else "Edible"

    return {"prediction": result, "confidence": float(preds[0])}


# Run the app with: uvicorn app:app --reload