#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preparar_corpus_lm08_streaming.py

• Procesa .txt grandes en streaming.
• Normaliza líneas según tokenizer.
• Filtra OOVs.
• Escribe directamente líneas válidas y descartadas.
"""

import sys, unicodedata, re
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer

# Argumentos
if len(sys.argv) < 2:
    sys.exit("❌ Pasa al menos un archivo .txt como argumento.")
INPUTS = [Path(p) for p in sys.argv[1:]]

TOKENIZER_DIR = "models/tokenizer_v2_base41"
OUT_CORPUS    = Path("data/lm/corpus_lm08_total.txt")
OUT_LOG       = Path("data/lm/descartes_oov_lm08.log")

# Tokenizer y vocabulario
print(f"📦 Cargando tokenizer desde: {TOKENIZER_DIR} …")
tok = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
vocab = set(tok.get_vocab().keys())
print(f"✅ Vocabulario cargado ({len(vocab)} tokens): {sorted(vocab)}")

# Normalizador
re_keep = re.compile(r"[a-zñüáéíóöú ]")

def normalize(line):
    line = line.lower()
    line = unicodedata.normalize("NFD", line)
    line = ''.join(c for c in line if unicodedata.category(c) != 'Mn')  # quita tildes
    line = re_keep.sub(lambda m: m.group(0), line)
    line = re.sub(r"\s+", " ", line).strip()
    return line.replace(" ", "|")

def is_in_vocab(line):
    return all(c in vocab for c in line)

# Salidas
OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
ok = OUT_CORPUS.open("w", encoding="utf-8")
bad = OUT_LOG.open("w", encoding="utf-8")

# Proceso en streaming
total, kept, discarded = 0, 0, 0

for file in INPUTS:
    print(f"\n📖 Procesando: {file}")
    with file.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            total += 1
            norm = normalize(line)
            if norm and is_in_vocab(norm):
                ok.write(norm + "\n")
                kept += 1
            else:
                bad.write(f"{file.name}:{i}:{norm}\n")
                discarded += 1
            if total % 100_000 == 0:
                print(f"  → {total:,} procesadas …")

# Final
ok.close()
bad.close()
print(f"\n✅ Finalizado: {kept:,} válidas | 🗑️ {discarded:,} descartadas → {OUT_CORPUS}")
