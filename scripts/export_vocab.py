# scripts/export_vocab.py
from transformers import Wav2Vec2Processor
import os

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
OUTPUT_PATH = "data/vocab/vocab.txt"

os.makedirs("data/vocab", exist_ok=True)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

vocab = processor.tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
tokens = [token for token, _ in vocab_sorted]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(tokens))

print("✅ vocab.txt generado:", OUTPUT_PATH)
