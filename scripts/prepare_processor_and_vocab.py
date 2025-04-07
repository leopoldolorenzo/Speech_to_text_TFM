# ✅ scripts/prepare_processor_and_vocab.py (actualizado con fix de serialización)

import os
import json
from datasets import load_from_disk
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

# === Configuración ===
DATASET_PATH = "data/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1"
VOCAB_FILE = os.path.join(TOKENIZER_DIR, "vocab.json")

os.makedirs(TOKENIZER_DIR, exist_ok=True)
print("[📂] Cargando dataset desde:", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)

# === Construir vocabulario ===
print("[🔤] Extrayendo caracteres únicos del dataset...")
all_text = " ".join(dataset["train"]["sentence"] + dataset["test"]["sentence"])
vocab_chars = sorted(list(set(all_text)))

vocab_dict = {c: i for i, c in enumerate(vocab_chars)}
vocab_dict["|"] = vocab_dict.pop(" ")  # espacio
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)

print(f"[✅] Vocabulario generado con {len(vocab_dict)} tokens.")
with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

# === Crear tokenizer y processor ===
tokenizer = Wav2Vec2CTCTokenizer(
    VOCAB_FILE,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
tokenizer.save_pretrained(TOKENIZER_DIR)

# Cargar processor base y asignar el tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor.tokenizer = tokenizer
processor.save_pretrained(TOKENIZER_DIR)

print("[💾] Tokenizer y Processor guardados en:", TOKENIZER_DIR)
print("[🏁] Preparación completada.")