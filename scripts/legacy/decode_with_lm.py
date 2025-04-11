# scripts/decode_with_lm.py
import sys
import numpy as np
from pyctcdecode import build_ctcdecoder

# === CONFIGURACIÓN ===
LOGITS_PATH = sys.argv[1]
VOCAB_PATH = "data/vocab/vocab.txt"
LM_PATH = "data/lm/modelo_limpio.bin"

# === CARGAR LOGITS ===
logits = np.load(LOGITS_PATH)

# === CARGAR VOCABULARIO ===
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_list = [line.strip() for line in f if line.strip()]

print(f"📚 Vocabulario cargado: {len(vocab_list)} tokens")

# === CONSTRUIR DECODER ===
decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path=LM_PATH
)

# === DECODIFICAR ===
print("🔡 Decodificando...")
transcription = decoder.decode(logits)

print("\n📝 Transcripción final:")
print(transcription)
