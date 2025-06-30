import json
import re
from pathlib import Path

# === Configuración de rutas ===
corpus_original = Path("data/lm/corpus_lm_v09.txt")
corpus_limpio = Path("data/lm/corpus_lm_v09_limpio.txt")
vocab_path = Path("models/tokenizer_v2_base41/vocab.json")

# === Cargar vocabulario ===
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Convertir vocab a conjunto de caracteres válidos
vocab_set = set(" " if k == "|" else k for k in vocab.keys())

# === Limpiar corpus ===
with open(corpus_original, "r", encoding="utf-8") as infile, open(corpus_limpio, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.lower()
        cleaned = ''.join(c if c in vocab_set else ' ' for c in line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            outfile.write(cleaned + "\n")

print(f"✅ Corpus limpio guardado en: {corpus_limpio}")
