# scripts/validate_labels.py — versión corregida

import re
from datasets import load_from_disk
from transformers import Wav2Vec2Processor

# === Configuración
DATASET_PATH = "data/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1"
MAX_UNK_THRESHOLD = 0.4

# === Cargar dataset completo
print("[📂] Cargando dataset...")
dataset = load_from_disk(DATASET_PATH)

# Eliminar la columna 'audio' para evitar decodificación
dataset = dataset.remove_columns(["audio"])

# === Cargar processor/tokenizer
print("[🔤] Cargando tokenizer limpio...")
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
unk_id = processor.tokenizer.unk_token_id

# === Limpieza de texto
def clean_text(text):
    return re.sub(r"[^a-záéíóúñü ]+", "", text.lower())

# === Análisis
print("\n🔍 Analizando primeras 10.000 muestras del dataset...\n")
total = 0
empty_labels = 0
many_unks = 0

for example in dataset["train"].select(range(10000)):
    total += 1
    cleaned = clean_text(example["sentence"])
    input_ids = processor.tokenizer(cleaned).input_ids

    if len(input_ids) == 0:
        empty_labels += 1
    else:
        unk_ratio = input_ids.count(unk_id) / len(input_ids)
        if unk_ratio > MAX_UNK_THRESHOLD:
            many_unks += 1

# === Resultados
print("📊 Resultado del análisis:")
print(f"🔢 Total ejemplos analizados : {total}")
print(f"🟥 Etiquetas vacías          : {empty_labels}")
print(f"🟨 Etiquetas con muchos [UNK]: {many_unks}")
print(f"🟩 Etiquetas válidas         : {total - empty_labels - many_unks}")
