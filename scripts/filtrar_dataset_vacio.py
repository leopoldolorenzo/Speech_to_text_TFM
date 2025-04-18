# filtrar_dataset_vacio.py
# Filtra ejemplos con transcripción vacía y guarda en nueva carpeta

from datasets import load_from_disk
from pathlib import Path

# === Configuración ===
DATASET_PATH = Path("data/common_voice_es/es/fine_tune_02/dataset_hf")
DATASET_OUT = Path("data/common_voice_es/es/fine_tune_02/dataset_hf_filtrado")

print(f"🔍 Cargando dataset desde: {DATASET_PATH}")
ds = load_from_disk(str(DATASET_PATH))

# === Filtrar ejemplos con transcripción vacía
print("🧹 Filtrando ejemplos con transcripción vacía...")
ds_filtrado = ds.filter(lambda x: x["sentence"].strip() != "")

# === Guardar en carpeta nueva
ds_filtrado.save_to_disk(str(DATASET_OUT))

print("✅ Dataset filtrado y guardado en:", DATASET_OUT)
print(f"📊 Nuevos tamaños: Train = {len(ds_filtrado['train'])} | Test = {len(ds_filtrado['test'])}")
