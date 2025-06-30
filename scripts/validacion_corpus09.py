import json
from pathlib import Path

# Ruta al archivo vocab.json
vocab_path = Path("models/tokenizer_v2_base41/vocab.json")

# Cargar el vocabulario
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Verificar cantidad total de tokens
total_tokens = len(vocab)
print(f"🔢 Total de tokens: {total_tokens}")

# Verificar existencia y posición de tokens especiales
expected_tokens = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4
}

print("\n🧩 Tokens especiales:")
for token, expected_index in expected_tokens.items():
    actual_index = vocab.get(token, None)
    status = "✅" if actual_index == expected_index else f"❌ (encontrado en {actual_index})"
    print(f"{token}: esperado {expected_index} → {status}")

# Mostrar ejemplo del vocabulario
print("\n📚 Ejemplo del vocabulario:")
for char, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"{idx:2}: {char}")
    if idx >= 20:  # solo mostrar los primeros 20 tokens
        break
