# verificar_dataset_entrenamiento.py
# Verifica que todos los audios del dataset están en formato .wav 16kHz y con transcripción válida

import os
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "data/common_voice_es/es/fine_tune_02/dataset_hf"
CLIPS_DIR = "data/common_voice_es/es/fine_tune_02/clips_wav"  # Base de los audios
TARGET_SR = 16000

print(f"🔍 Cargando dataset desde: {DATASET_DIR}")
dataset = load_from_disk(DATASET_DIR)

problemas = {
    "missing_files": [],
    "invalid_sr": [],
    "empty_text": [],
    "load_errors": []
}

def verificar(split_name):
    print(f"\n🔎 Verificando split: {split_name}")
    split = dataset[split_name]
    for ex in tqdm(split, desc=f"{split_name} examples"):
        ruta = ex["audio"]["path"]
        texto = ex["sentence"].strip()

        # 🛠️ Arreglar ruta si es relativa
        if not os.path.isabs(ruta):
            ruta = os.path.join(CLIPS_DIR, os.path.basename(ruta))

        if not os.path.exists(ruta):
            problemas["missing_files"].append(ruta)
            continue

        if not texto:
            problemas["empty_text"].append(ruta)

        try:
            _, sr = torchaudio.load(ruta)
            if sr != TARGET_SR:
                problemas["invalid_sr"].append((ruta, sr))
        except Exception as e:
            problemas["load_errors"].append((ruta, str(e)))

for split in ["train", "test"]:
    verificar(split)

print("\n📊 Informe final:")
print(f"❌ Archivos faltantes     : {len(problemas['missing_files'])}")
print(f"⚠️  Sampling rate incorrecto: {len(problemas['invalid_sr'])}")
print(f"⚠️  Transcripciones vacías : {len(problemas['empty_text'])}")
print(f"💥 Errores al cargar audio  : {len(problemas['load_errors'])}")

if any(len(lst) > 0 for lst in problemas.values()):
    print("\n🔧 Se encontraron problemas en el dataset. Revisa los detalles arriba.")
else:
    print("\n✅ Todo correcto. Dataset listo para entrenamiento.")
