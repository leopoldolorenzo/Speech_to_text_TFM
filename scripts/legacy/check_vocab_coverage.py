# check_vocab_coverage.py
import json

VOCAB_PATH = "models/tokenizer_v1_nospecial/vocab.json"
test_text = "Durante la contienda se afilió al partido comunista de españa"

# Limpiar texto como en entrenamiento
test_text = test_text.lower().strip()
test_text = "".join(c for c in test_text if c in "abcdefghijklmnopqrstuvwxyzáéíóúñü ")

# Cargar vocabulario
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Normalizar claves si es necesario
vocab_chars = set(vocab.keys())
if " " not in vocab_chars and "|" in vocab_chars:
    vocab_chars.add(" ")

# Verificar caracteres
missing = [c for c in set(test_text) if c not in vocab_chars]

print("✅ Caracteres en la frase:", sorted(set(test_text)))
print("🔎 Faltan en vocab.json:", missing if missing else "Ninguno 🚀")
