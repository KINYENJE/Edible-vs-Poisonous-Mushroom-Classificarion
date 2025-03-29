from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow CORS for all origins (adjust as needed for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = load_model(r"C:\Users\ADMIN\Desktop\mushroom_img_classification\models\mushroom_img_model.h5")

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