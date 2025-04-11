# scripts/create_subset.py

from datasets import load_from_disk, DatasetDict
from pathlib import Path

# === Configuración ===
SOURCE_PATH = "data/common_voice"
DEST_PATH = "data/common_voice_subset"
MAX_TRAIN = 30_000
MAX_TEST = 5_000

print("📂 Cargando dataset original...")
dataset = load_from_disk(SOURCE_PATH)

# Subset
print(f"✂️ Seleccionando {MAX_TRAIN} ejemplos de entrenamiento y {MAX_TEST} de test...")
subset = DatasetDict({
    "train": dataset["train"].select(range(min(MAX_TRAIN, len(dataset["train"])))),
    "test": dataset["test"].select(range(min(MAX_TEST, len(dataset["test"])))),
})

# Guardar
print(f"💾 Guardando subset en: {DEST_PATH}")
subset.save_to_disk(DEST_PATH)

# Confirmación
print(f"✅ Subset creado: {len(subset['train'])} train, {len(subset['test'])} test")
