# scripts/convertir_y_preparar_muestra_5.py

import os
import csv
import torchaudio
from pathlib import Path
from tqdm import tqdm

INPUT_TSV = "data/common_voice_es/es/validated.tsv"
INPUT_CLIPS = "data/common_voice_es/es/clips"
OUTPUT_DIR = "data/common_voice_es/es/fine_tune_05"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "clips_wav")
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "dataset_100k.tsv")

os.makedirs(OUTPUT_WAV, exist_ok=True)

# Cargar dataset TSV
print(f"📥 Cargando transcripciones desde: {INPUT_TSV}")
with open(INPUT_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    data = [row for row in reader if row["path"].strip() and row["sentence"].strip()]

print(f"📊 Total de muestras disponibles: {len(data)}")

# Seleccionar 100.000 ejemplos válidos
data = data[:100000]

# Procesar y exportar audios + transcripciones
print(f"🎧 Procesando y convirtiendo audios a WAV (100k ejemplos)...")
with open(OUTPUT_TSV, "w", encoding="utf-8") as out_tsv:
    writer = csv.writer(out_tsv, delimiter="\t")
    for row in tqdm(data):
        mp3_path = os.path.join(INPUT_CLIPS, row["path"])
        wav_name = row["path"].replace(".mp3", ".wav")
        wav_path = os.path.join(OUTPUT_WAV, wav_name)

        try:
            waveform, sr = torchaudio.load(mp3_path)
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            waveform = waveform.mean(dim=0, keepdim=True)  # convertir a mono
            torchaudio.save(wav_path, waveform, 16000)
        except Exception as e:
            print(f"⚠️ Error con {mp3_path}: {e}")
            continue

        # Limpiar texto
        texto = row["sentence"].lower().strip()
        texto = texto.replace("¿", "").replace("¡", "").replace(",", "").replace(".", "")
        texto = texto.replace(";", "").replace(":", "").replace("«", "").replace("»", "")
        texto = texto.replace("!", "").replace("?", "").replace("\"", "")

        writer.writerow([wav_path, texto])

print(f"✅ Dataset generado: {OUTPUT_TSV}")
print(f"🎧 Audios convertidos a: {OUTPUT_WAV}")


