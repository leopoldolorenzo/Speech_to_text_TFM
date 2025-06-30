#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preparar_corpus_lm08_streaming.py — versión optimizada

Uso:
    python scripts/preparar_corpus_lm08_streaming.py data/lm/es.txt
"""

import sys, unicodedata
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer

if len(sys.argv) != 2:
    sys.exit("❌ Uso: preparar_corpus_lm08_streaming.py <archivo.txt>")

CORPUS_IN = Path(sys.argv[1])
OUT_CORPUS = Path("data/lm/corpus_lm08_total.txt")
OUT_LOG = Path("data/lm/descartes_oov_lm08.log")
TOKENIZER_DIR = "models/tokenizer_v2_base41"

print(f"📦 Cargando tokenizer desde: {TOKENIZER_DIR} …")
tok = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
vocab = set(tok.get_vocab().keys())
print(f"✅ Vocabulario cargado ({len(vocab)} tokens): {sorted(vocab)}\n")

def normalize(line):
    line = line.lower()
    line = unicodedata.normalize("NFD", line)
    line = ''.join(c for c in line if unicodedata.category(c) != 'Mn')  # elimina tildes
    line = ''.join(c if c in 'abcdefghijklmnopqrstuvwxyzñüáéíóú ' else ' ' for c in line)
    line = ' '.join(line.split()).strip()
    return line.replace(" ", "|")

def is_valid(line):
    return all(char in vocab for char in line)

print(f"📖 Procesando {CORPUS_IN} …")
OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

n_valid, n_invalid = 0, 0

with CORPUS_IN.open(encoding="utf-8") as f_in, \
     OUT_CORPUS.open("w", encoding="utf-8") as f_out, \
     OUT_LOG.open("w", encoding="utf-8") as f_log:

    for i, raw in enumerate(f_in, 1):
        norm = normalize(raw)
        if norm and is_valid(norm):
            f_out.write(norm + "\n")
            n_valid += 1
        else:
            f_log.write(f"{i}:{norm}\n")
            n_invalid += 1

        if i % 100000 == 0:
            print(f"  → {i:,} líneas procesadas … ({n_valid:,} válidas)")

print("\n✅ Corpus preparado")
print(f"  Líneas válidas:     {n_valid:,}")
print(f"  Líneas descartadas: {n_invalid:,}")
print(f"  ➤ Guardado en:      {OUT_CORPUS}")
print(f"  ➤ Log descartes:    {OUT_LOG}")
