#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verificar_corpus_lm08.py

✅ Verifica que el corpus LM esté totalmente alineado con el tokenizador oficial tokenizer_v2_base41.
   - Coincidencia exacta de vocabulario (41 tokens).
   - Todas las líneas sólo con caracteres válidos.
   - Detección de duplicados o líneas vacías.

Uso:
    python scripts/verificar_corpus_lm08.py
"""

import json
import unicodedata
from pathlib import Path
from collections import Counter

# --- Configuración ---
VOCAB_PATH = Path("models/tokenizer_v2_base41/vocab.json")
CORPUS_PATH = Path("data/lm/corpus_lm08_total.txt")

# Tokens esperados (orden e índices importantes)
TOKENS_ESPERADOS = [
    "<pad>", "<s>", "</s>", "<unk>",  # Tokens especiales (índices 0–3)
    "|", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "á", "é", "í", "ñ", "ó", "ö", "ú", "ü", "'", "-"
]

# --- Cargar vocabulario ---
print(f"📦 Cargando vocabulario desde: {VOCAB_PATH}")
with VOCAB_PATH.open(encoding="utf-8") as f:
    vocab = json.load(f)

# Verifica número y orden
tokens_vocab = list(vocab.keys())
if tokens_vocab != TOKENS_ESPERADOS:
    print("❌ El vocabulario NO coincide con los 41 tokens esperados.")
    print("\nTokens esperados pero faltantes:")
    print(set(TOKENS_ESPERADOS) - set(tokens_vocab))
    print("\nTokens sobrantes o mal posicionados:")
    print(set(tokens_vocab) - set(TOKENS_ESPERADOS))
    exit(1)
else:
    print("✅ Vocabulario verificado (41 tokens, orden correcto).\n")

vocab_set = set(tokens_vocab)

# --- Verificación del corpus ---
print(f"🔍 Analizando corpus: {CORPUS_PATH}")
lineas_totales = 0
lineas_validas = 0
lineas_vacias = 0
lineas_oov = []
lineas_duplicadas = 0
lineas_unicas = set()
caracteres_oov = Counter()

with CORPUS_PATH.open(encoding="utf-8") as f:
    for num, linea in enumerate(f, 1):
        linea = unicodedata.normalize("NFC", linea.strip())
        lineas_totales += 1

        if not linea:
            lineas_vacias += 1
            continue

        if linea in lineas_unicas:
            lineas_duplicadas += 1
            continue
        lineas_unicas.add(linea)

        if all(ch in vocab_set for ch in linea):
            lineas_validas += 1
        else:
            oov_chars = [ch for ch in linea if ch not in vocab_set]
            caracteres_oov.update(oov_chars)
            lineas_oov.append((num, linea, oov_chars))

# --- Resultados ---
print("\n📊 RESULTADOS")
print("──────────────")
print(f"📝 Total líneas analizadas     : {lineas_totales:,}")
print(f"✅ Líneas válidas              : {lineas_validas:,}")
print(f"🗑️  Líneas vacías               : {lineas_vacias:,}")
print(f"🔁 Líneas duplicadas exactas   : {lineas_duplicadas:,}")
print(f"❌ Líneas con caracteres OOV   : {len(lineas_oov):,}")

if caracteres_oov:
    print("\n❗ Caracteres OOV detectados (top 10):")
    for char, freq in caracteres_oov.most_common(10):
        name = unicodedata.name(char, "UNKNOWN")
        print(f"  '{char}' (U+{ord(char):04X}) × {freq:,} → {name}")
else:
    print("✅ Sin caracteres OOV.")

# Guardar log opcional
LOG_PATH = CORPUS_PATH.with_name("verificacion_oov_lm08.log")
if lineas_oov:
    with LOG_PATH.open("w", encoding="utf-8") as logf:
        for num, linea, chars in lineas_oov:
            logf.write(f"Línea {num}: {linea}\n  OOV: {''.join(chars)}\n\n")
    print(f"\n📝 Log detallado guardado en: {LOG_PATH}")
else:
    print("🧼 Corpus 100% compatible con el tokenizador.")

