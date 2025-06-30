#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verificar_corpus_lm08_final.py

📌 Verifica que el corpus_lm08_total.txt:
• Solo contiene los 41 tokens del tokenizer.
• Los tokens están en el orden y posición correctos.
• Todas las líneas están correctamente normalizadas.
"""

import json
from pathlib import Path
import unicodedata

VOCAB_PATH = Path("models/tokenizer_v2_base41/vocab.json")
CORPUS_PATH = Path("data/lm/corpus_lm08_total.txt")

# Tokens esperados (en orden)
TOKENS_ESPERADOS = [
    "<pad>", "<s>", "</s>", "<unk>", "|", "'", "-", "a", "b", "c", "d", "e", "f", "g", "h",
    "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "á", "é", "í", "ñ", "ó", "ö", "ú", "ü"
]

# --- Verificación del vocabulario ---
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokens_actuales = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]

print("📦 Verificando orden y contenido del vocabulario…")
if tokens_actuales != TOKENS_ESPERADOS:
    print("❌ El vocabulario NO coincide con el esperado.\n")
    set_actual = set(tokens_actuales)
    set_esperado = set(TOKENS_ESPERADOS)

    faltan = set_esperado - set_actual
    sobran = set_actual - set_esperado

    if faltan:
        print(f"❌ FALTAN tokens: {faltan}")
    if sobran:
        print(f"❌ SOBRAN tokens: {sobran}")

    desordenados = [(i, t, a) for i, (t, a) in enumerate(zip(TOKENS_ESPERADOS, tokens_actuales)) if t != a]
    if desordenados:
        print("\n❗ Tokens en orden incorrecto:")
        for i, esperado, actual in desordenados:
            print(f"  Posición {i}: esperado='{esperado}' ≠ actual='{actual}'")

    exit(1)
else:
    print("✅ Vocabulario correcto (41 tokens en orden)\n")

# --- Verificación del corpus ---
print(f"📖 Verificando líneas en: {CORPUS_PATH} …")
fuera_vocab = {}
lineas_invalidas = 0
total = 0

caracteres_validos = set(tokens_actuales)

with CORPUS_PATH.open(encoding="utf-8") as f:
    for n, linea in enumerate(f, 1):
        total += 1
        linea = unicodedata.normalize("NFC", linea.strip())
        if not linea:
            continue
        for c in linea:
            if c not in caracteres_validos:
                fuera_vocab[c] = fuera_vocab.get(c, 0) + 1
                lineas_invalidas += 1
                break

print(f"\n📊 Resultado:")
print(f"  Líneas totales analizadas : {total:,}")
print(f"  Líneas inválidas (OOV)    : {lineas_invalidas:,}")
if fuera_vocab:
    print("\n❗ Caracteres fuera del vocabulario:")
    for char, freq in sorted(fuera_vocab.items(), key=lambda x: -x[1])[:20]:
        nombre = unicodedata.name(char, "UNKNOWN")
        print(f"  '{char}' (U+{ord(char):04X}) — {freq} veces — {nombre}")
else:
    print("✅ Todas las líneas están dentro del vocabulario.")

