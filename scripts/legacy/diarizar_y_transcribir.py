import os
import torch
import torchaudio
import numpy as np
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pyannote.audio import Pipeline

# === Configuración ===
ASR_MODEL = "models/fine_tune_03"  # Modelo ASR fine-tuneado
TOKENIZER_PATH = "models/fine_tune_03/vocab.json"  # Vocab compatible
LM_BIN_PATH = "models/lm_finetuneV02/modelo_finetuneV02.bin"
AUDIO_INPUT = "data/diarizacion/prueba01.wav"
TRANSCRIPCION_OUT = "data/diarizacion/diarizacion_transcripcion.txt"

# === Crear carpetas necesarias ===
os.makedirs("data/diarizacion", exist_ok=True)

# === Cargar modelo ASR y procesador ===
print("📦 Cargando modelo ASR + LM...")
processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL)

# === Extraer vocabulario desde vocab.json ===
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)
vocab_list = [k.replace(" ", "|") if k == " " else k for k, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
decoder = build_ctcdecoder(vocab_list, kenlm_model_path=LM_BIN_PATH)

# === Cargar pipeline de diarización desde disco ===
print("🗣️  Cargando pipeline de diarización local...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# === Ejecutar diarización ===
print(f"🔍 Analizando {AUDIO_INPUT}...")
diarization = pipeline(AUDIO_INPUT)

# === Transcribir segmentos por hablante ===
results = []
waveform, sr = torchaudio.load(AUDIO_INPUT)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start, end = int(turn.start * 16000), int(turn.end * 16000)
    segment = waveform[:, start:end]

    inputs = processor(segment.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
        transcription = decoder.decode(logits).strip().lower()

    results.append(f"{speaker}: {transcription}")

# === Guardar transcripción ===
with open(TRANSCRIPCION_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"✅ Diarización + transcripción guardada en {TRANSCRIPCION_OUT}")
