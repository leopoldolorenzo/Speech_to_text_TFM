# scripts/fix_audio_format_dataset.py

import os
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm

DATASET_PATH = "data/common_voice_subset"
FIXED_DATASET_PATH = "data/common_voice_subset_fixed"
TARGET_SR = 16000

# Asegurar directorio destino
os.makedirs(FIXED_DATASET_PATH, exist_ok=True)

def fix_audio_format(example):
    audio_info = example["audio"]
    original_path = audio_info["path"]

    if not os.path.exists(original_path):
        print(f"❌ Archivo no encontrado: {original_path}")
        return example

    try:
        waveform, sr = torchaudio.load(original_path)

        # Convertir si SR ≠ 16kHz o si es estéreo
        if sr != TARGET_SR or waveform.shape[0] > 1:
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            fixed_path = original_path.replace("common_voice_subset", "common_voice_subset_fixed")
            os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
            torchaudio.save(fixed_path, waveform, sample_rate=TARGET_SR)

            example["audio"]["path"] = fixed_path
        else:
            # Solo copiar si es correcto
            new_path = original_path.replace("common_voice_subset", "common_voice_subset_fixed")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            if not os.path.exists(new_path):
                torchaudio.save(new_path, waveform, sample_rate=sr)
            example["audio"]["path"] = new_path

    except Exception as e:
        print(f"💥 Error con {original_path}: {e}")

    return example

print(f"📂 Cargando dataset desde: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

print("🔍 Procesando audios...")
fixed_dataset = dataset.map(fix_audio_format)

print(f"💾 Guardando nuevo dataset en: {FIXED_DATASET_PATH}")
fixed_dataset.save_to_disk(FIXED_DATASET_PATH)
print("✅ Listo.")
