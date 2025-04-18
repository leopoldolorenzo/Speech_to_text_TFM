# compare_vocabularies.py

import json
from transformers import Wav2Vec2Processor

# === Configuración ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
CLEAN_VOCAB_PATH = "models/tokenizer_v1_nospecial/vocab.json"

# === Cargar vocabulario base del modelo original ===
print(f"🔍 Cargando vocabulario del modelo base: {BASE_MODEL}")
base_processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
base_vocab = set(base_processor.tokenizer.get_vocab().keys())

# === Cargar vocabulario limpio generado ===
print(f"🔍 Cargando vocabulario limpio: {CLEAN_VOCAB_PATH}")
with open(CLEAN_VOCAB_PATH, "r", encoding="utf-8") as f:
    clean_vocab = set(json.load(f).keys())

# === Comparación ===
only_in_base = sorted(base_vocab - clean_vocab)
only_in_clean = sorted(clean_vocab - base_vocab)
common = sorted(base_vocab & clean_vocab)

print("\n📌 Resultados de comparación:\n")
print(f"✅ Tokens comunes        : {len(common)}")
print(f"➖ Solo en modelo base   : {len(only_in_base)} → {only_in_base}")
print(f"➕ Solo en tokenizer limpio : {len(only_in_clean)} → {only_in_clean}")

print(f"\n📏 Tamaño total → base: {len(base_vocab)} | limpio: {len(clean_vocab)}")
