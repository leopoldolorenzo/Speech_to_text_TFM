from datasets import load_dataset, DatasetDict
from pathlib import Path

# === CONFIG ===
LANG = "es"
VERSION = "mozilla-foundation/common_voice_13_0"
OUTPUT_DIR = Path("data/common_voice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Cargar dataset completo ===
print("📥 Cargando dataset completo...")
dataset = load_dataset(VERSION, LANG, split="train+validation", use_auth_token=True)

# === 2. Filtrar audios sin texto o sin archivo ===
dataset = dataset.filter(lambda x: x["audio"] is not None and x["sentence"] is not None)

# === 3. Split 80/20 ===
print("🔀 Dividiendo en train/test (80/20)...")
split = dataset.train_test_split(test_size=0.2, seed=42)
dataset_split = DatasetDict({
    "train": split["train"],
    "test": split["test"]
})

# === 4. Guardar si se desea
print(f"💾 Guardando dataset en: {OUTPUT_DIR}")
dataset_split.save_to_disk(str(OUTPUT_DIR))

print("✅ Dataset listo para fine-tuning!")
# === 5. Guardar metadata ===
metadata = {
    "lang": LANG,
    "version": VERSION,
    "num_train_samples": len(dataset_split["train"]),
    "num_test_samples": len(dataset_split["test"]),
}
metadata_path = OUTPUT_DIR / "metadata.json"
with open(metadata_path, "w") as f:
    import json
    json.dump(metadata, f, indent=4)
print(f"💾 Metadata guardada en: {metadata_path}")
# === 6. Guardar info adicional ===
info_path = OUTPUT_DIR / "info.txt"
with open(info_path, "w") as f:
    f.write(f"Dataset: {VERSION}\n")
    f.write(f"Language: {LANG}\n")
    f.write(f"Train samples: {len(dataset_split['train'])}\n")
    f.write(f"Test samples: {len(dataset_split['test'])}\n")
print(f"💾 Info adicional guardada en: {info_path}")    
