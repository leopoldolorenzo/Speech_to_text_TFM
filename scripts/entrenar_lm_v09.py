#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
entrenar_lm09.py

📚 Entrena un modelo de lenguaje n-gramas (LM09) usando KenLM.
    - Usa corpus ya normalizado y compatible con el vocabulario del tokenizador.
    - Genera modelo ARPA y lo convierte a formato binario.
    - Usa fallback si hay problemas de descuentos.

Uso:
    python scripts/entrenar_lm09.py
"""

import subprocess
from pathlib import Path

# === Configuración ===
CORPUS_PATH = Path("data/lm/corpus_lm_v09_limpio.txt")
ARPA_PATH   = Path("models/lm_v09/modelo_lm_v09.arpa")
BIN_PATH    = Path("models/lm_v09/modelo_lm_v09.binary")
KENLM_BIN   = Path("tools/kenlm/build/bin/lmplz")        # ← ajusta si está en otra ruta
BUILD_BIN   = Path("tools/kenlm/build/bin/build_binary") # ← ajusta si está en otra ruta
NGRAM_ORDER = "5"

# === Verificaciones previas ===
print(f"📚 Corpus LM09: {CORPUS_PATH}")
if not CORPUS_PATH.is_file():
    raise FileNotFoundError(f"❌ No existe el corpus: {CORPUS_PATH}")

ARPA_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"📦 Guardando en: {ARPA_PATH.parent}")
print(f"🔢 N-gramas: {NGRAM_ORDER}\n")

# === Paso 1: Generar ARPA ===
print("⚙️  Generando modelo ARPA con KenLM …")

try:
    subprocess.run([
        str(KENLM_BIN),
        "-o", NGRAM_ORDER,
        "--text", str(CORPUS_PATH),
        "--arpa", str(ARPA_PATH),
        "--discount_fallback"
    ], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Error al ejecutar KenLM (lmplz): {e}")
    exit(1)

# === Paso 2: Convertir a binario ===
print("\n📦 Compilando modelo binario …")

try:
    subprocess.run([
        str(BUILD_BIN),
        str(ARPA_PATH),
        str(BIN_PATH)
    ], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Error al convertir a binario: {e}")
    exit(1)

# === Final ===
print("\n✅ Entrenamiento completado.")
print(f"📄 ARPA  → {ARPA_PATH}")
print(f"🧠 BIN   → {BIN_PATH}")
