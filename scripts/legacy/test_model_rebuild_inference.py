# test_model_rebuild_inference.py
# ------------------------------------------------------
# Prueba de inferencia para modelo entrenado Wav2Vec2.
# ------------------------------------------------------

import sys
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === CONFIGURACIÓN ===
MODEL_DIR = "models/spanish_w2v2_rebuild"
AUDIO_PATH = "data/eval/prueba01.wav"

print(f"🔍 Cargando modelo desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).eval()

# === Cargar audio
print(f"🔊 Cargando audio: {AUDIO_PATH}")
audio, sr = librosa.load(AUDIO_PATH, sr=16000)

# === Preprocesar entrada
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# === Inferencia
with torch.no_grad():
    logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)

# === Decodificar resultado
transcription = processor.batch_decode(pred_ids)[0]
print("\n📝 Transcripción:")
print(transcription)
