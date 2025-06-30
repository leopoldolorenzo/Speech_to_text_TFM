import os
import random
import shutil
import torchaudio
import csv
from pathlib import Path

# === Configuración ===
DIR_CLIPS = Path("data/common_voice_es/es/clips")
VALIDATED_TSV = Path("data/common_voice_es/es/validated.tsv")
DEST_WAV = Path("data/eval_confiable/wav")
DEST_TXT = Path("data/eval_confiable/txt")
N = 50  # cantidad de muestras

# === Crear carpetas destino
DEST_WAV.mkdir(parents=True, exist_ok=True)
DEST_TXT.mkdir(parents=True, exist_ok=True)

# === Seleccionar N archivos .mp3 aleatorios
mp3_files = list(DIR_CLIPS.glob("*.mp3"))
seleccionados = random.sample(mp3_files, N)

# === Leer transcripciones
print("📑 Cargando transcripciones desde validated.tsv...")
transcripciones = {}
with open(VALIDATED_TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        transcripciones[row["path"].replace(".mp3", "")] = row["sentence"]

# === Procesar cada archivo seleccionado
for mp3_path in seleccionados:
    base = mp3_path.stem  # sin extensión
    wav_dest = DEST_WAV / f"{base}.wav"
    txt_dest = DEST_TXT / f"{base}.txt"

    # === Convertir a WAV (mono, 16kHz)
    try:
        waveform, sr = torchaudio.load(mp3_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # convertir a mono
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform_16k = resampler(waveform)
        torchaudio.save(wav_dest.as_posix(), waveform_16k, 16000)
        print(f"🎧 {mp3_path.name} → {wav_dest.name}")
    except Exception as e:
        print(f"❌ Error al convertir {mp3_path.name}: {e}")
        continue

    # === Guardar transcripción
    texto = transcripciones.get(base, "")
    if texto:
        with open(txt_dest, "w", encoding="utf-8") as f:
            f.write(texto.lower())
    else:
        print(f"⚠️ No se encontró transcripción para: {base}")

print("\n✅ Evaluación lista en: data/eval_confiable/")
