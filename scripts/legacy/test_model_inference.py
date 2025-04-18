import sys
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Validar argumentos ===
assert len(sys.argv) == 3, "Uso: python test_model_inference.py <modelo_dir> <archivo_wav>"
MODEL_DIR = sys.argv[1]
AUDIO_PATH = sys.argv[2]

# === Cargar modelo y processor entrenado ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).eval()

# === Leer audio y preparar input ===
audio, _ = librosa.load(AUDIO_PATH, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# === Ejecutar inferencia ===
with torch.no_grad():
    logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)

# === Decodificar resultado ===
transcription = processor.batch_decode(pred_ids)[0]
print("📝 Transcripción:", transcription)
