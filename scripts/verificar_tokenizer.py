# verificar_tokenizer.py
# Verifica que el tokenizer y el modelo estén correctamente alineados

import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Configuración ===
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"
MODEL_PATH = "models/fine_tune_02"  # Cambia si usas otro modelo

print(f"📦 Cargando processor desde: {TOKENIZER_DIR}")
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
vocab = processor.tokenizer.get_vocab()

print("🔤 Tokens en vocabulario:", len(vocab))
print("🔣 Tokens especiales:")
print(" - pad_token:", repr(processor.tokenizer.pad_token))
print(" - bos_token:", repr(processor.tokenizer.bos_token))
print(" - eos_token:", repr(processor.tokenizer.eos_token))

if os.path.exists(MODEL_PATH):
    print(f"\n🧠 Cargando modelo desde: {MODEL_PATH}")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    lm_head_size = model.lm_head.out_features
    print(f"🔎 lm_head.out_features = {lm_head_size}")

    if lm_head_size != len(vocab):
        print("❌ ERROR: El vocabulario y el lm_head tienen diferente tamaño!")
    else:
        print("✅ Tamaños coinciden. Modelo y tokenizer alineados.")
else:
    print("⚠️ No se ha encontrado el modelo en disco. Solo se verificó el tokenizer.")