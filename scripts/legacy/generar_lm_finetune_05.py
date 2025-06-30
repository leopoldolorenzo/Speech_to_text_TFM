# scripts/generar_lm_finetune_05.py

import os
import csv
import json
import subprocess

# Rutas y archivos
VOCAB_PATH = "models/tokenizer_v2_base41/vocab.json"
VALIDATED_TSV = "data/common_voice_es/es/validated.tsv"
CORPUS_TXT = "data/lm/corpus_finetune_05.txt"
ARPA_PATH = "models/lm_finetuneV05/modelo_finetune05.arpa"
BIN_PATH = "models/lm_finetuneV05/modelo_finetune05.bin"

# Crear carpetas si no existen
os.makedirs("data/lm", exist_ok=True)
os.makedirs("models/lm_finetuneV05", exist_ok=True)

# Cargar vocabulario de 41 tokens
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = set(json.load(f).keys())

def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = texto.replace(" ", "|")
    texto = ''.join([c for c in texto if c in vocab])
    return texto

# Extraer frases desde validated.tsv
print(f"📥 Procesando transcripciones desde: {VALIDATED_TSV}")
frases_validas = []

with open(VALIDATED_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        frase = limpiar_texto(row["sentence"])
        if frase:
            frases_validas.append(frase)

print(f"✅ Total de frases válidas: {len(frases_validas)}")

# Guardar corpus limpio
with open(CORPUS_TXT, "w", encoding="utf-8") as f:
    for linea in frases_validas:
        f.write(linea + "\n")

print(f"💾 Corpus limpio guardado en: {CORPUS_TXT}")

# Generar modelo .arpa
print("⚙️ Generando modelo ARPA con KenLM...")
cmd_arpa = f"~/TFM/tools/kenlm-bin/lmplz -o 5 < {CORPUS_TXT} > {ARPA_PATH}"
subprocess.run(cmd_arpa, shell=True, executable="/bin/bash")

# Compilar modelo binario .bin
print("📦 Compilando modelo binario...")
cmd_bin = f"~/TFM/tools/kenlm-bin/build_binary -q 8 -b 8 -a 22 -T /tmp {ARPA_PATH} {BIN_PATH}"
subprocess.run(cmd_bin, shell=True, executable="/bin/bash")

print("✅ Modelo de lenguaje generado con éxito:")
print(f" - ARPA: {ARPA_PATH}")
print(f" - BIN : {BIN_PATH}")
