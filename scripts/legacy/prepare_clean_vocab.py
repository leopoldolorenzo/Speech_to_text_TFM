# prepare_clean_vocab.py
import os
import json
import re
from datasets import load_from_disk

DATASET_PATH = "data/common_voice_subset"
OUTPUT_DIR = "models/tokenizer_v1_nospecial"
VOCAB_FILE = os.path.join(OUTPUT_DIR, "vocab.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[📂] Cargando dataset desde:", DATASET_PATH)
dataset = load_from_disk(DATASET_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü ]+", "", text)
    return text

print("[🧼] Limpiando texto...")
all_text = " ".join([clean_text(t) for t in dataset["train"]["sentence"]] + [clean_text(t) for t in dataset["test"]["sentence"]])
vocab = sorted(set(all_text))

vocab_dict = {c: i for i, c in enumerate(vocab)}
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict.pop(" ")
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"[✅] Vocabulario guardado en: {VOCAB_FILE} con {len(vocab_dict)} tokens.")
