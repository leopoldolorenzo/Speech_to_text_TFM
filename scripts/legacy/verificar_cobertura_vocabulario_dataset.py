import json
import csv
import unicodedata

# Configuración de rutas
VOCAB_PATH = "models/tokenizer_v2_base41/vocab.json"
DATASET_TSV = "data/common_voice_es/es/fine_tune_07/dataset_150k.tsv"

# Cargar vocabulario
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)
vocab_chars = set(vocab_dict.keys())

# Estadísticas
total_lineas = 0
lineas_ok = 0
lineas_vacias = 0
con_espacios = []
con_oov = []
con_invisibles = []

# Revisión línea por línea
with open(DATASET_TSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for n, row in enumerate(reader):
        if len(row) != 2:
            print(f"[⚠️] Línea {n}: formato incorrecto: {row}")
            continue

        audio_path, texto = row
        total_lineas += 1

        # 1. Vacía
        if not texto.strip():
            lineas_vacias += 1
            continue

        # 2. Espacios reales (deben ser '|')
        if ' ' in texto:
            con_espacios.append((n, texto))

        # 3. Invisibles o de control (excepto |)
        invisibles = [c for c in texto if unicodedata.category(c)[0] == 'C' and c != '|']
        if invisibles:
            con_invisibles.append((n, texto, invisibles))

        # 4. Caracteres fuera de vocabulario
        oov = [c for c in texto if c not in vocab_chars]
        if oov:
            con_oov.append((n, texto, oov))

        if not oov and not ' ' in texto and not invisibles:
            lineas_ok += 1

# Resultados
print("\n📊 Resultado del chequeo completo:\n")
print(f"✔️ Líneas procesadas       : {total_lineas:,}")
print(f"✔️ Líneas correctas        : {lineas_ok:,}")
print(f"❌ Transcripciones vacías  : {lineas_vacias}")
print(f"❌ Con espacios reales (' '): {len(con_espacios)}")
print(f"❌ Con caracteres invisibles: {len(con_invisibles)}")
print(f"❌ Con OOV fuera del vocab  : {len(con_oov)}\n")

# Muestras de errores
if con_espacios:
    print("🔎 Ejemplo con espacio real:")
    print(f"Línea {con_espacios[0][0]}: {con_espacios[0][1]}\n")

if con_oov:
    print("🔎 Ejemplo con caracteres OOV:")
    n, t, o = con_oov[0]
    print(f"Línea {n}: {t}")
    print(f"→ Caracteres OOV: {o}\n")

if con_invisibles:
    print("🔎 Ejemplo con invisibles/control:")
    n, t, inv = con_invisibles[0]
    print(f"Línea {n}: {t}")
    print(f"→ Caracteres invisibles: {[f'U+{ord(c):04X}' for c in inv]}\n")

# Recomendación
if lineas_ok == total_lineas:
    print("✅ El dataset está limpio y listo para el entrenamiento.\n")
else:
    print("⚠️ Se recomienda depurar las líneas con problemas antes de entrenar.\n")
