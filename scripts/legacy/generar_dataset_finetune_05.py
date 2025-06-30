# scripts/generar_dataset_finetune_05.py

import os
import csv
from datasets import Dataset, DatasetDict, Audio, load_dataset
from sklearn.model_selection import train_test_split

TSV_PATH = "data/common_voice_es/es/fine_tune_05/dataset_100k.tsv"
OUTPUT_DIR = "data/common_voice_es/es/fine_tune_05/dataset_hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar TSV
print(f"📥 Cargando dataset desde: {TSV_PATH}")
data = []

with open(TSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) == 2:
            path, text = row
            data.append({"audio": path, "text": text})

print(f"📊 Total de ejemplos cargados: {len(data)}")

# Split 80% train / 20% validation
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Crear DatasetDict
print("📦 Construyendo DatasetDict con datasets.Audio...")
ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
})

# Castear columna audio como tipo especial
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Guardar en disco
print(f"💾 Guardando dataset en: {OUTPUT_DIR}")
ds.save_to_disk(OUTPUT_DIR)

print("✅ Dataset HuggingFace creado y guardado exitosamente.")
