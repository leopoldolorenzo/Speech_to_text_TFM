from transformers import Wav2Vec2Processor
import sys

if len(sys.argv) < 2:
    print("Uso: python test_tokenizer.py <ruta_modelo>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]

print(f"🔍 Cargando processor desde: {MODEL_PATH}")

# === Cargar processor ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# === Texto de prueba ===
texto = "hola qué tal estás"

# === Tokenización ===
with processor.as_target_processor():
    tokens = processor.tokenizer(texto).input_ids

# === Resultados ===
print("\n[🧪 PRUEBA TOKENIZER]")
print(f"📝 Texto: {texto}")
print(f"🧩 Token IDs: {tokens}")
print(f"🔤 Tokens decodificados: {processor.tokenizer.decode(tokens)}")
