# scripts/verificar_decoder_lm_finetune05.py

"""
Verificación de compatibilidad entre el modelo de lenguaje KenLM (.bin)
y el vocabulario original (41 tokens) del tokenizer usado en fine_tune_05.
"""

from pyctcdecode import build_ctcdecoder
import json

# Ruta al vocabulario del tokenizer exportado desde el modelo base
VOCAB_PATH = "models/tokenizer_v2_base41/vocab.json"

# Ruta al modelo de lenguaje binario compilado
LM_BIN_PATH = "models/lm_finetuneV05/modelo_finetune05.bin"

# Cargar vocabulario y extraer etiquetas
print(f"📥 Cargando vocabulario desde: {VOCAB_PATH}")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_list = list(json.load(f).keys())

# Verificar número de tokens
print(f"🔠 Número total de tokens: {len(vocab_list)}")

# Construir el decodificador
print(f"🧠 Construyendo decoder con modelo LM en: {LM_BIN_PATH}")
decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path=LM_BIN_PATH
)

print("✅ Decodificador construido correctamente. El modelo de lenguaje es compatible con el vocabulario.")
