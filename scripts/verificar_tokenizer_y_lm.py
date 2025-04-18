# verificar_tokenizer_y_lm.py
# --------------------------------------------------
# Verifica la coherencia entre el tokenizer y el modelo de lenguaje KenLM
# --------------------------------------------------

import os
from transformers import Wav2Vec2Processor

# === RUTAS ===
TOKENIZER_PATH = "models/tokenizer_v1_nospecial"
ARPA_PATH = "data/lm/modelo_finetune.arpa"

# === 1. Cargar el processor (tokenizer)
print(f"📦 Cargando tokenizer desde: {TOKENIZER_PATH}")
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_PATH)

# === 2. Revisar los tokens
vocab = processor.tokenizer.get_vocab()
tokens = list(vocab.keys())

print(f"🔤 Total de tokens en vocabulario: {len(tokens)}")
print("📌 Tokens del vocabulario:")
print(tokens)

if " " in tokens:
    print("⚠️ El token de espacio ' ' está en el vocabulario (esto no es lo esperado).")
else:
    print("✅ El token de espacio ' ' NO está en el vocabulario (correcto).")

# === 3. Probar codificación y decodificación
texto = "el gato come"
ids = processor.tokenizer(texto).input_ids
reconstruido = processor.tokenizer.decode(ids, skip_special_tokens=True)

print("\n🧪 Prueba de codificación/decodificación:")
print("🔸 Texto original    :", texto)
print("🔹 IDs codificados   :", ids)
print("🔁 Reconstruido      :", reconstruido)

if reconstruido.replace(" ", "") == texto.replace(" ", ""):
    print("✅ El tokenizer reconstruye correctamente sin introducir errores.")
else:
    print("⚠️ La reconstrucción tiene diferencias.")

# === 4. Verificar si el modelo KenLM contiene n-gramas con espacios
print(f"\n📂 Verificando modelo LM: {ARPA_PATH}")
if not os.path.exists(ARPA_PATH):
    print("❌ Archivo .arpa no encontrado.")
else:
    with open(ARPA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        encontrados = 0
        for linea in f:
            if " " in linea and not linea.startswith("\\"):
                encontrados += 1
                if encontrados <= 5:
                    print(f"🔎 N-grama con espacio: {linea.strip()}")
        if encontrados:
            print(f"✅ Se encontraron {encontrados} líneas con espacios (correcto).")
        else:
            print("⚠️ No se encontraron n-gramas con espacios. Verifica el corpus.")
