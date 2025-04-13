
from feature_extraction.feat_extract import extract_features, convert_mp3_to_wav

convert_mp3_to_wav()
features = extracted_features = extract_features()

from fastapi import FastAPI
import torch
from training.train_dnn import LSTM_EmotionClassifier

input_size = 41
hidden_size = 128
num_classes = 4
app = FastAPI()

# Load model
model = LSTM_EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("emotion_model_weights.pth"))
model.eval()

@app.post("/predict")
async def predict(features: List[float]):
    input_tensor = torch.tensor(features).reshape(1, 1, -1).float()
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return {"emotion": int(prediction)}


