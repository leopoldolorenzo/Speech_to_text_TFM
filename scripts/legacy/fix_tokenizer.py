from transformers import Wav2Vec2Processor
import os

MODEL_DIR = "models/tokenizer_v1_nospecial"

print(f"🔧 Cargando processor desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

# ❌ Eliminar tokens que sobran
processor.tokenizer.bos_token = None
processor.tokenizer.eos_token = None

# 🧼 Limpiar el added_tokens_decoder
ids_to_remove = []
for idx, entry in processor.tokenizer.added_tokens_decoder.items():
    if entry.content in ["<s>", "</s>"]:
        ids_to_remove.append(idx)

for idx in ids_to_remove:
    del processor.tokenizer.added_tokens_decoder[idx]

# ✅ Guardar processor limpio
processor.save_pretrained(MODEL_DIR)
print(f"[✅] Tokenizer limpiado y guardado en: {MODEL_DIR}")
