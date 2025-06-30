# generar_dataset_finetune_02.py
# Prepara un DatasetDict HuggingFace desde un TSV generado con muestras .wav convertidas (fine_tune_02)

import os
import csv
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm

# === Rutas ===
DIR_BASE = "data/common_voice_es/es/fine_tune_02"
TSV_IN = os.path.join(DIR_BASE, "dataset_100k.tsv")
OUT_PATH = os.path.join(DIR_BASE, "dataset_hf")

# === Cargar y preparar ejemplos desde el TSV ===
data = []
with open(TSV_IN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in tqdm(reader, desc="📋 Cargando TSV"):
        data.append({
            "audio": row["file"],
            "sentence": row["transcription"]
        })

# === Dividir dataset (80% train, 20% test)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

# === Crear DatasetDict
ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test": Dataset.from_list(test_data)
})

# === Formatear columna de audio
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

# === Guardar como dataset HuggingFace
ds.save_to_disk(OUT_PATH)

print(f"\n✅ Dataset HuggingFace preparado y guardado en: {OUT_PATH}")
print(f"📊 Tamaños: Train = {len(ds['train'])} | Test = {len(ds['test'])}")
