import os
import random
import torchaudio
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# === Configuración ===
SRC_TSV = Path("data/common_voice_es/es/validated.tsv")
SRC_CLIPS = Path("data/common_voice_es/es/clips")
DST_DIR = Path("data/eval_confiable_100/wav")
TXT_DIR = Path("data/eval_confiable_100/txt")
N = 100
SEED = 42

# === Crear carpetas destino ===
DST_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

# === Leer TSV y seleccionar aleatoriamente N muestras ===
df = pd.read_csv(SRC_TSV, sep="\t")
df = df[df["path"].notnull() & df["sentence"].notnull()]
df = df.sample(n=N, random_state=SEED).reset_index(drop=True)

# === Procesar y convertir audios ===
print(f"🎯 Convirtiendo {N} archivos MP3 → WAV 16kHz mono...")

errores = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    mp3_path = SRC_CLIPS / row["path"]
    wav_name = f"{i:03d}.wav"
    txt_name = f"{i:03d}.txt"

    try:
        # Cargar y convertir
        waveform, sr = torchaudio.load(mp3_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform_16k = resampler(waveform)

        # Guardar WAV
        torchaudio.save(DST_DIR / wav_name, waveform_16k, 16000)

        # Guardar TXT
        with open(TXT_DIR / txt_name, "w", encoding="utf-8") as f:
            f.write(row["sentence"].strip())

    except Exception as e:
        errores.append((mp3_path, str(e)))

# === Resumen ===
print(f"✅ Proceso completado. Archivos guardados en: {DST_DIR} y {TXT_DIR}")
if errores:
    print(f"⚠️ {len(errores)} errores durante la conversión.")
    with open("errores_conversion_eval_100.log", "w") as f:
        for path, err in errores:
            f.write(f"{path}: {err}\n")
