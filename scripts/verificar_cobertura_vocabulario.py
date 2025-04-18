# verificar_cobertura_vocabulario.py
# Verifica si el vocabulario del tokenizer cubre todos los caracteres del dataset

from datasets import load_from_disk
from transformers import Wav2Vec2Processor
from collections import Counter
import os

# === Configuración ===
DATASET_PATH = "data/common_voice_es/es/fine_tune_02/dataset_hf"
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"

# === Cargar tokenizer y dataset
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
vocab = list(processor.tokenizer.get_vocab().keys())

dataset = load_from_disk(DATASET_PATH)

# === Extraer todos los caracteres del dataset
def extraer_caracteres(dataset):
    chars = Counter()
    for split in ["train", "test"]:
        for example in dataset[split]:
            texto = example["sentence"].lower().strip()
            chars.update(list(texto))
    return chars

caracteres_dataset = extraer_caracteres(dataset)

# === Comparar con el vocabulario
tokens_relevantes = set("".join([t for t in vocab if len(t) == 1 and t.isalpha() or t == " "]))
caracteres_unicos = set(caracteres_dataset.keys())

# === Resultados
print("\n✅ Tokens en vocabulario:", sorted(tokens_relevantes))
print("📊 Caracteres encontrados en dataset:", sorted(caracteres_unicos))

faltantes = caracteres_unicos - tokens_relevantes
extra = tokens_relevantes - caracteres_unicos

print("\n🚨 Caracteres en el dataset que NO están en el vocabulario:", sorted(faltantes))
print("📎 Tokens en el vocabulario que NO aparecen en el dataset:", sorted(extra))

print("\n📈 Caracteres más frecuentes en el dataset:")
print(caracteres_dataset.most_common(20))

