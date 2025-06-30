#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verificar_dataset_preparado.py

Valida que todas las transcripciones de un dataset TSV estén completamente limpias
y se ajusten exactamente al vocabulario de 41 tokens del tokenizer del modelo base.

Uso:
    python verificar_dataset_preparado.py \
        --tsv data/common_voice_es/es/fine_tune_08/dataset_100k.tsv \
        --vocab models/tokenizer_v2_base41/vocab.json
"""

import csv
import json
import sys
import argparse
from pathlib import Path
from collections import Counter

# ---------- ARGUMENTOS CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--tsv", required=True, type=Path, help="Ruta al archivo TSV a verificar")
parser.add_argument("--vocab", required=True, type=Path, help="Ruta al vocab.json del tokenizer")
args = parser.parse_args()

# ---------- CARGA VOCABULARIO ----------
try:
    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
        vocab_chars = set(vocab.keys())
except FileNotFoundError:
    sys.exit(f"❌ No se encontró el vocabulario en {args.vocab}")
except json.JSONDecodeError:
    sys.exit(f"❌ Error leyendo el vocabulario: {args.vocab} no es JSON válido")

# ---------- VERIFICACIÓN ----------
total_lineas = 0
oov_counter = Counter()
lineas_oov = []

with open(args.tsv, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, row in enumerate(reader, 1):
        total_lineas += 1

        if len(row) != 2:
            print(f"⚠️  Línea {i} malformada (esperadas 2 columnas): {row}")
            continue

        _, text = row
        for ch in text:
            if ch not in vocab_chars:
                oov_counter[ch] += 1
                lineas_oov.append((i, text))
                break

# ---------- RESULTADOS ----------
print("\n📊 Resultados de la verificación:")
print(f"• Total líneas procesadas: {total_lineas}")
print(f"• Líneas válidas:          {total_lineas - len(lineas_oov)}")
print(f"• Líneas con OOV:          {len(lineas_oov)}")

if not oov_counter:
    print("\n✅ El dataset está completamente limpio y es compatible con el vocabulario.")
else:
    print("\n❌ Se encontraron caracteres fuera de vocabulario (OOV).")
    print("🔣 Caracteres OOV más frecuentes:")
    for ch, count in oov_counter.most_common():
        print(f"  {repr(ch)} → {count} veces")

    print("\n🧾 Ejemplos de líneas con OOV:")
    for i, txt in lineas_oov[:5]:
        print(f"  Línea {i}: {txt}")

    print("\n❗ Revisa y corrige el dataset antes de entrenar.")
