# convertir_y_preparar_muestra_4.py
# Convierte una muestra de 200.000 archivos nuevos .mp3 a .wav 16kHz mono,
# evitando repetir los usados en fine_tune_01, _02 y _03. Genera TSV limpio.

import os
import csv
import re
import torchaudio
from tqdm import tqdm

# === CONFIGURACIÓN ===
DIR_BASE = "data/common_voice_es/es"
CLIPS_MP3 = os.path.join(DIR_BASE, "clips")
CLIPS_WAV = os.path.join(DIR_BASE, "fine_tune_04", "clips_wav")
TSV_ORIG = os.path.join(DIR_BASE, "validated.tsv")
TSV_OUT = os.path.join(DIR_BASE, "fine_tune_04", "dataset_200k.tsv")
USED_LIST_1 = os.path.join(DIR_BASE, "fine_tune_01", "pistas_usadas_1.txt")
USED_LIST_2 = os.path.join(DIR_BASE, "fine_tune_02", "pistas_usadas_2.txt")
USED_LIST_3 = os.path.join(DIR_BASE, "fine_tune_03", "pistas_usadas_3.txt")
USED_LIST_4 = os.path.join(DIR_BASE, "fine_tune_04", "pistas_usadas_4.txt")
TARGET_SR = 16000
MAX_EJEMPLOS = 200000

# === Preparación de carpetas ===
os.makedirs(CLIPS_WAV, exist_ok=True)
os.makedirs(os.path.dirname(TSV_OUT), exist_ok=True)

# === Cargar pistas ya usadas ===
usadas = set()
for path in [USED_LIST_1, USED_LIST_2, USED_LIST_3]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            usadas.update(line.strip() for line in f)

# === Función para limpiar texto ===
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúñü ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# === PROCESAMIENTO ===
with open(TSV_ORIG, "r", encoding="utf-8") as fin, \
     open(TSV_OUT, "w", encoding="utf-8", newline="") as fout, \
     open(USED_LIST_4, "w", encoding="utf-8") as flog:

    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.writer(fout, delimiter="\t")
    writer.writerow(["file", "transcription"])

    errores = 0
    convertidos = 0

    for row in tqdm(reader, desc="🔄 Procesando muestras (evitando duplicados)"):
        if convertidos >= MAX_EJEMPLOS:
            break

        nombre_mp3 = row["path"]
        texto = clean_text(row["sentence"])

        if not nombre_mp3 or not texto or nombre_mp3 in usadas:
            continue

        ruta_mp3 = os.path.join(CLIPS_MP3, nombre_mp3)
        nombre_wav = nombre_mp3.replace(".mp3", ".wav")
        ruta_wav = os.path.join(CLIPS_WAV, nombre_wav)

        if not os.path.exists(ruta_mp3):
            errores += 1
            continue

        try:
            if not os.path.exists(ruta_wav):
                audio, sr = torchaudio.load(ruta_mp3)
                if sr != TARGET_SR:
                    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                    audio = resample(audio)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                torchaudio.save(ruta_wav, audio, sample_rate=TARGET_SR)

            writer.writerow([ruta_wav, texto])
            flog.write(nombre_mp3 + "\n")
            convertidos += 1

        except Exception as e:
            print(f"\n❌ Error procesando {nombre_mp3}: {e}")
            errores += 1

# === RESUMEN ===
print("\n✅ Proceso completado.")
print(f"🎧 Audios convertidos correctamente: {convertidos}")
print(f"⚠️ Errores detectados: {errores}")
print(f"📄 Dataset TSV generado en: {TSV_OUT}")
print(f"🗒️  Pistas nuevas usadas guardadas en: {USED_LIST_4}")
