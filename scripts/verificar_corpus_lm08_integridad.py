#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verificar_corpus_lm08_integridad.py

✅ Verifica que el vocabulario contiene exactamente 41 tokens válidos.
✅ Verifica que el corpus usa exclusivamente esos tokens.
✅ Reporta cualquier carácter fuera de vocabulario.
✅ Verifica integridad general del corpus.

Uso:
    python scripts/verificar_corpus_lm08_integridad.py
"""

import json
from pathlib import Path
from collections import Counter
import unicodedata

# --- Configuración ---
VOCAB_PATH = Path("models/tokenizer_v2_base41/vocab.json")
CORPUS_PATH = Path("data/lm/corpus_lm08_total.txt")

# --- Cargar vocabulario ---
print(f"📦 Cargando vocabulario desde: {VOCAB_PATH}")
with VOCAB_PATH.open(encoding="utf-8") as f:
    vocab_json = json.load(f)

# Extraer caracteres reales del vocab
vocab_chars = set()
for token in vocab_json:
    if len(token) == 1:
        vocab_chars.add(token)

print(f"🔠 Tokens únicos encontrados: {len(vocab_chars)}")
if len(vocab_chars) != 41:
    print(f"❌ El vocabulario no tiene exactamente 41 tokens (encontró {len(vocab_chars)})")
else:
    print("✅ El vocabulario contiene exactamente 41 caracteres.")

# --- Verificar corpus ---
print(f"\n📖 Verificando corpus: {CORPUS_PATH}")
total = 0
errores = 0
lineas_vacias = 0
caracteres_oov = Counter()

with CORPUS_PATH.open(encoding="utf-8") as f:
    for i, linea in enumerate(f, 1):
        linea = unicodedata.normalize("NFC", linea.strip())

        if not linea:
            lineas_vacias += 1
            continue

        for ch in linea:
            if ch not in vocab_chars:
                errores += 1
                caracteres_oov[ch] += 1

        total += 1
        if i % 500000 == 0:
            print(f"  → {i:,} líneas procesadas …")

# --- Resultados ---
print(f"\n📊 RESULTADOS")
print(f"────────────────────────────")
print(f"📌 Líneas procesadas:    {total:,}")
print(f"⚠️  Líneas vacías:        {lineas_vacias:,}")
print(f"❗ Caracteres OOV:        {errores:,}")

if caracteres_oov:
    print("\n❗ Top 20 caracteres fuera del vocabulario:")
    for ch, count in caracteres_oov.most_common(20):
        nombre = unicodedata.name(ch, "UNKNOWN")
        print(f"  '{ch}' (U+{ord(ch):04X}): {count} veces – {nombre}")
else:
    print("✅ No se encontraron caracteres fuera del vocabulario.")

if errores == 0 and lineas_vacias == 0 and len(vocab_chars) == 41:
    print("\n✅ El corpus es 100% compatible con tokenizer_v2_base41.")
else:
    print("\n⚠️  El corpus presenta inconsistencias. Revisa los detalles anteriores.")
