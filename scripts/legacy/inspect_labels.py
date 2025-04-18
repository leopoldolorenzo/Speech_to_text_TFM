from datasets import load_from_disk
from transformers import Wav2Vec2Processor
import os

DATASET_PATH = "data/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1"

dataset = load_from_disk(DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

print("🔍 Mostrando ejemplos del dataset preprocesado:")
for i in range(5):
    item = dataset["train"][i]
    text = item.get("sentence", "[no sentence]")
    labels = item.get("labels", None)

    print(f"\n📝 Original: {text}")
    if labels is None:
        print("⚠️ No se encontraron etiquetas en este ejemplo.")
    else:
        print(f"🔢 Labels   : {labels}")
        print(f"🔡 Decodificadas: {processor.tokenizer.decode(labels)}")
