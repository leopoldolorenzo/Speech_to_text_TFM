# generar_lm_finetune_04.py
# Filtra el corpus y genera un modelo de lenguaje ARPA y BIN compatible con fine_tune_04

import re
import subprocess
from pathlib import Path
from tqdm import tqdm

# === Paths ===
corpus_in = Path("data/lm/corpus_final.txt")                     # Corpus base
corpus_out = Path("data/lm/corpus_finetune_04.txt")              # Corpus filtrado para fine_tune_04
arpa_path = Path("models/lm_finetune_04/modelo_finetune_04.arpa")# LM ARPA
bin_path = Path("models/lm_finetune_04/modelo_finetune_04.bin")  # LM BIN

# === Vocabulario permitido (del tokenizer del modelo fine-tune_04)
caracteres_validos = set("abcdefghijklmnopqrstuvwxyzáéíóúñü ")

# === Crear carpeta destino si no existe
bin_path.parent.mkdir(parents=True, exist_ok=True)

# === Paso 1: Filtrar corpus para que coincida con el vocabulario
print("🔤 Filtrando corpus para vocabulario del modelo...")

total_lineas = sum(1 for _ in corpus_in.open("r", encoding="utf-8"))

with corpus_in.open("r", encoding="utf-8") as fin, corpus_out.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, total=total_lineas, desc="🧹 Filtrando"):
        line = line.strip().lower()
        line = re.sub(r"[^a-záéíóúñü\s]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and all(c in caracteres_validos for c in line):
            fout.write(line + "\n")

print(f"📚 Corpus limpio guardado en: {corpus_out}")

# === Paso 2: Generar modelo ARPA con KenLM
print("🧠 Generando modelo ARPA con KenLM...")
subprocess.run([
    "tools/kenlm-bin/lmplz",
    "--text", str(corpus_out),
    "--arpa", str(arpa_path),
    "--order", "5",
    "--prune", "0", "0", "1"  # Opcional: reduce n-gramas 5-gram poco frecuentes
], check=True)

print(f"📄 Modelo ARPA guardado en: {arpa_path}")

# === Paso 3: Compilar modelo BIN
print("📦 Compilando modelo BIN optimizado...")
subprocess.run([
    "tools/kenlm-bin/build_binary",
    str(arpa_path),
    str(bin_path)
], check=True)

print(f"✅ Modelo BIN listo en: {bin_path}")
