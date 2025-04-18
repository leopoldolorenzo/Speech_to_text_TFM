# debug_model_logits.py
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np

MODEL_DIR = "models/debug_w2v2"
AUDIO_PATH = "data/eval/prueba01.wav"

# === Cargar modelo y processor ===
print(f"📦 Cargando modelo desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).eval()

# === Cargar audio ===
print(f"🔊 Cargando audio: {AUDIO_PATH}")
audio, sr = librosa.load(AUDIO_PATH, sr=16000)

# === Preprocesamiento ===
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# === Inferencia ===
with torch.no_grad():
    logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)

# === Resultados ===
print("\n🔢 Predicted token IDs:")
print(pred_ids[0].tolist())

print("\n🔤 Texto decodificado (greedy):")
decoded = processor.batch_decode(pred_ids)[0]
print(decoded)
