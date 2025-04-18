# inspect_tokenizer_vocab.py
import sys
from transformers import Wav2Vec2Processor
from pprint import pprint
import os

assert len(sys.argv) > 1, "📂 Debes indicar la ruta del modelo como argumento"
MODEL_DIR = sys.argv[1]

print(f"\n🔍 [inspect_tokenizer_vocab.py] Cargando tokenizer desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer = processor.tokenizer

print("\n📏 Tamaño del vocabulario:", len(tokenizer))
print("\n🔤 Tokens especiales definidos:")
print("  PAD:", tokenizer.pad_token, f"({tokenizer.pad_token_id})")
print("  UNK:", tokenizer.unk_token, f"({tokenizer.unk_token_id})")
print("  BOS:", tokenizer.bos_token, f"({tokenizer.bos_token_id})")
print("  EOS:", tokenizer.eos_token, f"({tokenizer.eos_token_id})")

print("\n🔎 Tokens 'sospechosos' en vocabulario:")
vocab = tokenizer.get_vocab()
sospechosos = [t for t in vocab if t.startswith("<") or t.startswith("[") or t == "</s>"]
pprint(sospechosos)
