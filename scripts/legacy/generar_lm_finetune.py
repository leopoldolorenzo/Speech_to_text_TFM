# generar_lm_finetune.py
# Este script filtra un corpus grande para adecuarlo al vocabulario del modelo fine-tuneado
# y luego genera un modelo de lenguaje (LM) en formato ARPA y BIN usando KenLM.

import re
import subprocess
from pathlib import Path
from tqdm import tqdm

# === Paths ===
corpus_in = Path("data/lm/corpus_final.txt")                  # Corpus original completo
corpus_out = Path("data/lm/corpus_finetune.txt")              # Corpus filtrado
arpa_path = Path("data/lm/modelo_finetune.arpa")              # LM intermedio en formato ARPA
bin_path = Path("data/lm/modelo_finetune.bin")                # LM final binario para pyctcdecode

# === Vocabulario permitido (coincide con vocab.json del modelo fine-tuneado)
caracteres_validos = set("abcdefghijklmnopqrstuvwxyzáéíóúñü ")

# === Paso 1: Filtrar corpus para que coincida con el vocabulario
print("🔤 Filtrando corpus para que coincida con el vocabulario del modelo fine-tuneado...")

# Contamos líneas para barra de progreso
total_lineas = sum(1 for _ in corpus_in.open("r", encoding="utf-8"))

with corpus_in.open("r", encoding="utf-8") as fin, corpus_out.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, total=total_lineas, desc="🧹 Filtrando", unit="línea"):
        line = line.strip().lower()
        line = re.sub(r"[^a-záéíóúñü\s]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and all(c in caracteres_validos for c in line):
            fout.write(line + "\n")

print(f"📚 Corpus limpio guardado en: {corpus_out}")

# === Paso 2: Generar modelo ARPA con KenLM
print("🧠 Generando modelo ARPA con KenLM...")
lmplz = "tools/kenlm-bin/lmplz"

subprocess.run([
    lmplz,
    "--text", str(corpus_out),
    "--arpa", str(arpa_path),
    "--order", "5",
    "--prune", "0", "0", "1"  # Elimina n-gramas poco frecuentes para ahorrar espacio
], check=True)

print(f"📄 Archivo ARPA generado en: {arpa_path}")

# === Paso 3: Compilar modelo binario
print("📦 Compilando modelo BIN optimizado...")
build_bin = "tools/kenlm-bin/build_binary"

subprocess.run([build_bin, str(arpa_path), str(bin_path)], check=True)

print(f"✅ Modelo BIN generado correctamente en: {bin_path}")
