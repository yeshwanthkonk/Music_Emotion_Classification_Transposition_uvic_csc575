import os
import json
import time

import torch
import mimetypes
from fastapi import FastAPI, UploadFile, File, Form
from training.train_dnn import LSTM_EmotionClassifier
from pre_processing.new_mp3_to_midi import transpose
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse, JSONResponse
from feature_extraction.feat_extract import extract_features, convert_mp3_to_wav, prepare_features_for_prediction


input_size = 41
hidden_size = 128
num_classes = 4

# Load model
model = LSTM_EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("./pretrained_models/emotion_model_weights.pth"))
model.eval()
label_map = {0: "happy", 1: "angry", 2: "sad", 3: "calm"}

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/process/")
async def process_music(file: UploadFile = File(...), target_emotion: str = Form(...)):
    mp3_file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(mp3_file_path, "wb") as f:
        f.write(await file.read())

    wav_path = mp3_file_path.replace(".mp3", ".wav")
    convert_mp3_to_wav(mp3_file_path, wav_path)
    features = extract_features(wav_path)
    features = prepare_features_for_prediction(*features)
    input_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    predicted_emotion = label_map[prediction]

    if predicted_emotion.lower() == target_emotion.lower():
        return JSONResponse({
            "message": "No transposition needed",
            "emotion": predicted_emotion,
            "file_url": f"/download/{file.filename}"
        })

    modified_path = transpose(mp3_file_path, predicted_emotion, target_emotion)

    modified_name = os.path.basename(modified_path)
    return JSONResponse({
        "detected_emotion": predicted_emotion,
        "file_url": f"/download/{modified_name}"
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
