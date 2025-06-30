#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

CORPUS_PATH = Path("data/lm/corpus_lm08_total.txt")
VOCAB_PATH = Path("models/tokenizer_v2_base41/vocab.json")

# Cargar vocabulario del tokenizer
with VOCAB_PATH.open(encoding="utf-8") as f:
    vocab = json.load(f)

tokens_validos = {t for t in vocab if len(t) == 1}  # Ignora tokens especiales (<s>, <pad>, etc.)

print(f"📦 Vocabulario cargado: {len(tokens_validos)} tokens válidos")
print(f"🔤 Tokens válidos: {sorted(tokens_validos)}")

# Extraer caracteres únicos del corpus
caracteres_corpus = set()
with CORPUS_PATH.open(encoding="utf-8") as f:
    for linea in f:
        caracteres_corpus.update(set(linea.strip()))

print(f"\n📄 Tokens únicos en el corpus: {len(caracteres_corpus)}")
print(f"🔡 Tokens encontrados: {sorted(caracteres_corpus)}")

# Verificaciones
fuera_vocab = caracteres_corpus - tokens_validos
no_usados = tokens_validos - caracteres_corpus

print("\n🧪 Verificación:")
if fuera_vocab:
    print(f"❌ Caracteres FUERA del vocabulario ({len(fuera_vocab)}): {sorted(fuera_vocab)}")
else:
    print("✅ No hay caracteres fuera del vocabulario.")

if no_usados:
    print(f"⚠️  Tokens del vocabulario NO usados en el corpus ({len(no_usados)}): {sorted(no_usados)}")
else:
    print("✅ Todos los tokens del vocabulario fueron usados al menos una vez.")

# Resultado final
if not fuera_vocab and not no_usados:
    print("\n🎯 El corpus cubre perfectamente el vocabulario de 41 tokens y no contiene caracteres extraños.")
else:
    print("\n⚠️  El corpus NO cumple completamente con la cobertura esperada.")
