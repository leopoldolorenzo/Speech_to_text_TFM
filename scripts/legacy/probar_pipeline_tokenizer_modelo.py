# probar_pipeline_tokenizer_modelo.py (corregido)
# Verifica que el processor y el modelo entrenado generan logits válidos y se puede decodificar correctamente

import torch
import numpy as np
import random
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Configuración ===
DATASET_DIR = "data/common_voice_es/es/fine_tune_01/dataset_hf"
MODEL_DIR = "models/fine_tune_01"
#MODEL_DIR = "training/fine_tune_01/checkpoint-3750"


# === Cargar processor y modelo
print(f"📦 Cargando processor desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

print(f"📦 Cargando modelo entrenado desde: {MODEL_DIR}")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

# === Cargar dataset y elegir muestra aleatoria
dataset = load_from_disk(DATASET_DIR)
sample = random.choice(dataset["test"])
audio_input = sample["audio"]["array"]
ground_truth = sample["sentence"]

# === Procesar audio
inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# === Decodificación
pred_ids = torch.argmax(logits, dim=-1)
pred_text_default = processor.batch_decode(pred_ids)[0]
pred_text_nogroup = processor.batch_decode(pred_ids, group_tokens=False)[0]

# === Mostrar resultados
print("\n🔍 Ejemplo de prueba")
print(f"REF (Transcripción real): {ground_truth}")
print(f"HYP (default decoding):  {pred_text_default}")
print(f"HYP (no group_tokens):   {pred_text_nogroup}")
print(f"🔢 Tokens predichos:      {pred_ids[0].tolist()}")
