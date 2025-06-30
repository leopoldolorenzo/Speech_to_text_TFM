import os
import json
import subprocess

# Rutas y archivos
VOCAB_PATH = "models/tokenizer_v2_base41/vocab.json"
CORPUS_TXT_ORIG = "data/lm/corpus.txt"          # Corpus original sin limpiar
CORPUS_TXT_LIMPIO = "data/lm/corpus_finetune_06.txt"
CORPUS_TXT_DEDUP = "data/lm/corpus_finetune_06_dedup.txt"
ARPA_PATH = "models/lm_finetuneV06/modelo_finetune06.arpa"
BIN_PATH = "models/lm_finetuneV06/modelo_finetune06.bin"

# Crear carpetas si no existen
os.makedirs("data/lm", exist_ok=True)
os.makedirs("models/lm_finetuneV06", exist_ok=True)

# Cargar vocabulario 41 tokens
print("🔹 Cargando vocabulario...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = set()
    tokens = json.load(f)
    for token in tokens.keys():
        if token == " ":
            vocab.add("|")
        else:
            for c in token:
                vocab.add(c)

print(f"Caracteres válidos ({len(vocab)}): {''.join(sorted(vocab))}")

# Función para limpiar línea
def limpiar_texto(texto):
    texto = texto.lower().strip().replace(" ", "|")
    return "".join([c for c in texto if c in vocab])

# Limpiar corpus original
print(f"\n📥 Limpiando corpus original: {CORPUS_TXT_ORIG}")
lineas_leidas = 0
lineas_escritas = 0
with open(CORPUS_TXT_ORIG, "r", encoding="utf-8") as fin, \
     open(CORPUS_TXT_LIMPIO, "w", encoding="utf-8") as fout:
    for line in fin:
        lineas_leidas += 1
        cleaned = limpiar_texto(line)
        if cleaned:
            fout.write(cleaned + "\n")
            lineas_escritas += 1
        if lineas_leidas % 1000000 == 0:
            print(f"  - Procesadas {lineas_leidas:,} líneas, escritas {lineas_escritas:,} líneas")

print(f"✅ Corpus limpio generado: {lineas_escritas} líneas válidas de {lineas_leidas}")

# Deduplicar corpus limpio para evitar errores en KenLM
print(f"\n🔄 Deduplicando corpus limpio para evitar errores en KenLM...")
sort_cmd = f"sort {CORPUS_TXT_LIMPIO} | uniq > {CORPUS_TXT_DEDUP}"
res = subprocess.run(sort_cmd, shell=True)
if res.returncode != 0:
    print("❌ Error al deduplicar corpus.")
    exit(1)
print(f"✅ Corpus deduplicado guardado en {CORPUS_TXT_DEDUP}")

# Generar modelo ARPA con KenLM, con fallback para evitar BadDiscountException
print(f"\n⚙️ Generando modelo ARPA 5-gramas con KenLM (con discount_fallback)...")
cmd_arpa = f"~/TFM/tools/kenlm-bin/lmplz --discount_fallback -o 5 < {CORPUS_TXT_DEDUP} > {ARPA_PATH}"
res = subprocess.run(cmd_arpa, shell=True, executable="/bin/bash")
if res.returncode != 0:
    print("❌ Error durante la generación del modelo ARPA.")
    exit(1)
print(f"✅ Modelo ARPA generado en {ARPA_PATH}")

# Compilar modelo binario .bin optimizado para pyctcdecode
print(f"\n📦 Compilando modelo binario optimizado (.bin)...")
cmd_bin = f"~/TFM/tools/kenlm-bin/build_binary -q 8 -b 8 -a 22 -T /tmp {ARPA_PATH} {BIN_PATH}"
res = subprocess.run(cmd_bin, shell=True, executable="/bin/bash")
if res.returncode != 0:
    print("❌ Error durante la compilación del modelo binario.")
    exit(1)
print(f"✅ Modelo binario compilado en {BIN_PATH}")

print("\n🎉 Proceso completado exitosamente.")
print(f"Archivos generados:\n - Corpus limpio: {CORPUS_TXT_LIMPIO}\n - Corpus dedup: {CORPUS_TXT_DEDUP}\n - Modelo ARPA: {ARPA_PATH}\n - Modelo BIN: {BIN_PATH}")
