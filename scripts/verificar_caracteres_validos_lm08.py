# verificar_caracteres_validos_lm08.py
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

CORPUS_PATH = Path("data/lm/corpus_lm08_total.txt")
VOCAB_PATH = Path("models/tokenizer_v2_base41/vocab.json")
SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>"}  # los que no aparecerán en texto

print(f"📦 Cargando vocabulario desde: {VOCAB_PATH}")
with VOCAB_PATH.open("r", encoding="utf-8") as f:
    vocab = json.load(f)

# Solo los tokens que son visibles en el corpus
caracteres_validos = {tok for tok in vocab if len(tok) == 1 and tok not in SPECIAL_TOKENS}

print(f"✅ Caracteres válidos esperados ({len(caracteres_validos)}): {sorted(caracteres_validos)}")

print(f"\n🔍 Analizando archivo: {CORPUS_PATH} …")
invalid_chars = Counter()
n_lineas = 0

with CORPUS_PATH.open("r", encoding="utf-8") as f:
    for linea in tqdm(f, desc="⏳ Verificando líneas"):
        n_lineas += 1
        for char in linea.strip():
            if char not in caracteres_validos:
                invalid_chars[char] += 1

print(f"\n📄 Líneas procesadas: {n_lineas:,}")
if invalid_chars:
    print(f"\n❌ Caracteres inválidos encontrados:")
    for c, freq in invalid_chars.most_common(20):
        print(f"  '{c}' (U+{ord(c):04X}) → {freq} veces")
else:
    print("\n✅ Todas las líneas usan únicamente los caracteres válidos.")
