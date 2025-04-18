# save_checkpoint_as_model.py
# ------------------------------------------------------
# Guarda el checkpoint más reciente como modelo final.
# ------------------------------------------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

CHECKPOINT_PATH = "training/spanish_w2v2_rebuild/checkpoint-1000"
FINAL_MODEL_DIR = "models/spanish_w2v2_rebuild"
TOKENIZER_PATH = "models/tokenizer_v1_nospecial"

print(f"🔄 Cargando checkpoint desde: {CHECKPOINT_PATH}")
model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_PATH)

print(f"💾 Guardando modelo en: {FINAL_MODEL_DIR}")
model.save_pretrained(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)

print("✅ Modelo final guardado con éxito.")
