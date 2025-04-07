# scripts/extract_logits.py
import torch
import librosa
import numpy as np
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

if len(sys.argv) < 3:
    print("Uso: python extract_logits.py entrada.wav salida.npy")
    sys.exit(1)

audio_path = sys.argv[1]
output_path = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval().to(device)

audio, _ = librosa.load(audio_path, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits.squeeze(0).cpu().numpy()

np.save(output_path, logits)
print("✅ Logits guardados en:", output_path)

