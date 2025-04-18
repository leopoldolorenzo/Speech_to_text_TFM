# validate_processor_and_dataset.py
# ----------------------------------------
# Valida processor, tokenizer y dataset:
# - Verifica vocabulario
# - Revisa ejemplos del dataset
# - Simula decodificación desde logits aleatorios
# ----------------------------------------

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === CONFIG ===
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"
DATASET_PATH = "data/common_voice_subset"
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

print(f"🔍 Cargando processor desde: {TOKENIZER_DIR}")
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
tokenizer = processor.tokenizer

print(f"\n📏 Tamaño del vocabulario: {len(tokenizer)}")
print("🆗 Tokens clave:")
for t in ["[PAD]", "[UNK]", "|"]:
    print(f"  {t}: ID = {tokenizer.convert_tokens_to_ids(t)}")

print("\n📚 Cargando dataset desde:", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)
sample = dataset["train"][0]

# === Revisar ejemplo ===
print("\n🔎 Ejemplo de texto limpio y etiquetas:")
text = sample["sentence"]
input_audio = sample["audio"]["array"]

print("Texto original:", text)
with processor.as_target_processor():
    label_ids = processor.tokenizer(text.lower()).input_ids
print("Label IDs:", label_ids)

# === Simular decodificación ===
print("\n🎲 Simulando logits aleatorios y decodificando...")
fake_logits = torch.randn(1, 50, len(tokenizer))  # [batch, time, vocab]
pred_ids = torch.argmax(fake_logits, dim=-1)
decoded = processor.batch_decode(pred_ids)
print("Decodificado:", decoded[0])

# === Comprobación final ===
if len(label_ids) == 0:
    print("⚠️ Etiquetas vacías. Verificá el tokenizer.")
elif all(id == -100 for id in label_ids):
    print("⚠️ Todos los labels son -100. Algo anda mal.")
else:
    print("✅ Todo parece OK para entrenar.")
