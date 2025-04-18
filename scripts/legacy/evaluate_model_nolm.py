# scripts/evaluate_model_nolm.py
# -----------------------------------------------------------------------------
# Este script evalúa un modelo Wav2Vec2 (sin modelo de lenguaje) usando un
# conjunto de validación definido en un archivo TSV.
# 
# Para cada archivo de audio:
#   - Carga el audio y lo procesa con el processor del modelo.
#   - Realiza inferencia con el modelo (sin usar decoder de lenguaje).
#   - Calcula el WER (Word Error Rate) y CER (Character Error Rate).
#
# Los resultados se imprimen por consola y se guardan en un archivo CSV.
#
# Uso:
#   python scripts/evaluate_model_nolm.py nombre_modelo
# Ejemplo:
#   python scripts/evaluate_model_nolm.py spanish_w2v2_rebuild
# -----------------------------------------------------------------------------


import os
import sys
import pandas as pd
from jiwer import wer, cer
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === CONFIGURACIÓN ===
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL_DIR = f"models/{MODEL_NAME}"
TSV_PATH = "data/val_dataset.tsv"
OUTPUT_CSV = f"results/metrics_{MODEL_NAME}_nolm.csv"
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Cargar processor y modelo
print(f"📦 Cargando modelo: {MODEL_NAME}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()

# === Verificar dataset
assert os.path.exists(TSV_PATH), f"❌ No se encontró: {TSV_PATH}"
df = pd.read_csv(TSV_PATH, sep="\t", header=0)
print(f"🔍 Evaluando {len(df)} audios...\n")

# === Evaluación
results = []
for _, row in df.iterrows():
    audio_path = row["file"]
    reference = row["transcription"]

    try:
        # Cargar y procesar audio
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hypothesis = processor.batch_decode(pred_ids)[0]

    except Exception as e:
        print(f"❌ Error en {audio_path}: {e}")
        continue

    wer_score = wer(reference, hypothesis)
    cer_score = cer(reference, hypothesis)

    print(f"📄 {os.path.basename(audio_path)}")
    print(f"REF: {reference}")
    print(f"HYP: {hypothesis}")
    print(f"WER: {wer_score:.3f} | CER: {cer_score:.3f}\n")

    results.append({
        "audio_file": audio_path,
        "reference": reference,
        "hypothesis": hypothesis,
        "WER": wer_score,
        "CER": cer_score
    })

# === Guardar resultados
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Resultados guardados en {OUTPUT_CSV}")
print("🔚 Evaluación finalizada.")
