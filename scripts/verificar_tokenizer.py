#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preparar_corpus_lm08.py

• Lee varios .txt (uno por línea).
• Normaliza (minúsculas, elimina acentos, borra puntuación, colapsa espacios).
• Reemplaza espacio → "|" (separador que usa CTC).
• Filtra líneas con OOVs respecto a los 41 tokens.
• Evita duplicados.
• Guarda corpus limpio y log de descartes.

Uso:
    python preparar_corpus_lm08.py \
           data/lm/es.txt \
           data/lm/corpus_OpenSubtitles2024.txt
"""
import sys, unicodedata, re, os
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer

# ----- args -----
if len(sys.argv) < 2:
    sys.exit("❌  Pasa al menos un archivo .txt como argumento.")
INPUTS = [Path(p) for p in sys.argv[1:]]

TOKENIZER_DIR = "models/tokenizer_v2_base41"
OUT_CORPUS    = Path("data/lm/corpus_lm08_total.txt")
OUT_LOG       = Path("data/lm/descartes_oov_lm08.log")

# ----- tokenizer & vocab -----
tok = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)
vocab = set(tok.get_vocab().keys())   # 41 símbolos (incluye '|')

# ----- normalización -----
re_keep = re.compile(r"[a-zñüáéíóú ]")   # tras quitar tildes diacríticas

def normalize(line: str) -> str:
    line = line.lower()
    line = unicodedata.normalize("NFD", line)
    line = ''.join(c for c in line if unicodedata.category(c) != 'Mn')  # quita tildes
    line = re_keep.sub(lambda m: m.group(0), line)                     # limpia
    line = re.sub(r"\s+", " ", line).strip()
    return line.replace(" ", "|")

def is_in_vocab(line: str) -> bool:
    return all(ch in vocab for ch in line)

# ----- proceso -----
valid, discards = set(), []

for txt in INPUTS:
    if not txt.is_file():
        print(f"⚠️  {txt} no existe, se ignora.")
        continue
    print(f"📖  Leyendo {txt} …")
    with txt.open() as f:
        for i, raw in enumerate(f, 1):
            norm = normalize(raw)
            if norm and is_in_vocab(norm):
                valid.add(norm)
            else:
                discards.append(f"{txt.name}:{i}:{norm}")

# ----- guardar -----
OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
OUT_CORPUS.write_text("\n".join(sorted(valid)), encoding="utf-8")
OUT_LOG.write_text("\n".join(discards), encoding="utf-8")

print(f"\n✅  Corpus limpio: {len(valid):,} líneas  →  {OUT_CORPUS}")
print(f"🗑️   Descartes   : {len(discards):,} líneas  →  {OUT_LOG}")
