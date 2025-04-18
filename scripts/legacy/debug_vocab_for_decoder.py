# debug_vocab_for_decoder.py
from transformers import Wav2Vec2Processor
from pprint import pprint

MODEL_DIR = "models/debug_w2v2"

print(f"📦 Cargando processor desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

vocab_dict = processor.tokenizer.get_vocab()
# ordenado por índice (no por clave)
vocab = sorted(vocab_dict, key=vocab_dict.get)

print(f"🔤 Tokens pasados a pyctcdecode (total: {len(vocab)}):\n")
pprint(vocab)

# Detectar tokens inválidos
invalid = [t for t in vocab if len(t) != 1 and t not in ("[PAD]", "[UNK]")]
if invalid:
    print("\n❌ Tokens inválidos detectados:")
    pprint(invalid)
else:
    print("\n✅ Todos los tokens tienen formato válido (1 char o especiales).")
