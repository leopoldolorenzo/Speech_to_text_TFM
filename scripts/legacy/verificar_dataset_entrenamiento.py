# scripts/verificar_dataset_entrenamiento.py

import os
import csv
import torchaudio
from tqdm import tqdm

TSV_PATH = "data/common_voice_es/es/fine_tune_05/dataset_100k.tsv"
errores = []

print(f"🔍 Verificando integridad del dataset: {TSV_PATH}")

with open(TSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, row in enumerate(tqdm(reader)):
        if len(row) != 2:
            errores.append((i, "Fila incompleta"))
            continue

        wav_path, texto = row
        if not os.path.isfile(wav_path):
            errores.append((i, f"Archivo no encontrado: {wav_path}"))
            continue

        if texto.strip() == "":
            errores.append((i, "Transcripción vacía"))
            continue

        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                errores.append((i, f"Frecuencia incorrecta: {sr} Hz"))
            if waveform.shape[0] != 1:
                errores.append((i, "Audio no es mono"))
        except Exception as e:
            errores.append((i, f"Error al cargar audio: {e}"))

print("\n✅ Verificación completada.")
if errores:
    print(f"⚠️ Se detectaron {len(errores)} problemas:")
    for idx, error in errores[:10]:
        print(f" - Línea {idx}: {error}")
    if len(errores) > 10:
        print("  ...")
else:
    print("🎉 Todos los archivos están en buen estado.")
