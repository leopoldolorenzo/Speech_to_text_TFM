import unicodedata
import langdetect
import json
from collections import Counter
from tqdm import tqdm
import re

# --- CONFIGURACIÓN ---
CORPUS_PATH = 'data/lm/corpus_lm07_total.txt'
VOCAB_PATH = 'models/tokenizer_v2_base41/vocab.json'
N_MUESTRA = 50000
VOCAB_ESPERADO = set("abcdefghijklmnopqrstuvwxyzáéíóúñü¿¡ ")

# --- FUNCIONES ---
def cargar_vocabulario(path_vocab):
    with open(path_vocab, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    caracteres = set()
    for token in vocab.keys():
        if len(token) == 1:
            caracteres.add(token)
    return caracteres

def analizar_linea(linea):
    tokens = linea.strip().split()
    long_token = len(tokens)
    chars_fuera = [c for c in linea if c not in vocabulario and not c.isspace()]
    return long_token, chars_fuera

def detectar_idioma(linea):
    try:
        return langdetect.detect(linea)
    except:
        return 'err'

# --- MAIN ---
print("⏳ Cargando vocabulario...")
vocabulario = cargar_vocabulario(VOCAB_PATH)
print(f"[✓] Vocabulario cargado con {len(vocabulario)} caracteres únicos")

# Verifica si el vocabulario cubre los 41 esperados
faltantes = VOCAB_ESPERADO - vocabulario
sobrantes = vocabulario - VOCAB_ESPERADO
if faltantes:
    print(f"⚠️ FALTAN caracteres esperados: {sorted(faltantes)}")
if sobrantes:
    print(f"⚠️ SOBRAN caracteres no esperados: {sorted(sobrantes)}")
if not faltantes and not sobrantes:
    print("✅ Vocabulario coincide perfectamente con los 41 esperados.")

# Inicializa contadores
conteo_longs = []
caracteres_fuera_vocab = Counter()
idiomas = Counter()
lineas_vistas = set()
duplicados = 0

print(f"\n🔍 Analizando primeras {N_MUESTRA} líneas del corpus...\n")

with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    for i, linea in enumerate(tqdm(f, total=N_MUESTRA, desc="Procesando líneas")):
        if i >= N_MUESTRA:
            break
        linea = unicodedata.normalize('NFC', linea)
        if linea in lineas_vistas:
            duplicados += 1
        else:
            lineas_vistas.add(linea)

        long_tok, chars_out = analizar_linea(linea)
        conteo_longs.append(long_tok)
        caracteres_fuera_vocab.update(chars_out)
        idioma = detectar_idioma(linea)
        idiomas[idioma] += 1

# --- RESULTADOS ---
print("\n📊 RESULTADOS DEL ANÁLISIS")
print("────────────────────────────")
print(f"📌 Líneas analizadas:        {len(conteo_longs)}")
print(f"🔁 Líneas duplicadas exactas: {duplicados}")
print(f"🗣️ Idiomas detectados (top 5): {idiomas.most_common(5)}")
print(f"🧮 Longitud (tokens):        min={min(conteo_longs)}, max={max(conteo_longs)}, media={sum(conteo_longs)//len(conteo_longs)}")

if caracteres_fuera_vocab:
    print("\n❗ Caracteres fuera del vocabulario (top 20):")
    for char, freq in caracteres_fuera_vocab.most_common(20):
        nombre = unicodedata.name(char, 'UNKNOWN')
        print(f"  '{char}' (U+{ord(char):04X}): {freq} veces – {nombre}")
else:
    print("\n✅ Todos los caracteres están dentro del vocabulario.")
