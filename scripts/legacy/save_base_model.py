# scripts/save_base_model.py
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
OUTPUT_DIR = "models/base"

print(f"📥 Descargando modelo base desde: {MODEL_ID}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

print(f"💾 Guardando modelo en: {OUTPUT_DIR}")
processor.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)

# Guardar el nombre del modelo original
with open(os.path.join(OUTPUT_DIR, "huggingface_model.txt"), "w") as f:
    f.write(MODEL_ID + "\n")

print("📝 Nombre del modelo original guardado en huggingface_model.txt")
print("✅ Modelo base guardado correctamente.")
