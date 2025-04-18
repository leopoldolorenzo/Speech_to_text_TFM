# scripts/corregir_audios_dataset.py

import os
import torchaudio
from datasets import load_from_disk, DatasetDict, Audio

INPUT_DATASET = "data/common_voice_subset"
OUTPUT_DATASET = "data/common_voice_limpio"
TARGET_SR = 16000

print(f"[📦] Cargando dataset desde: {INPUT_DATASET}")
dataset = load_from_disk(INPUT_DATASET)

errores = []
total_convertidos = 0

def fix_audio(batch):
    global total_convertidos
    path = batch["audio"]["path"]
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as e:
        errores.append(path)
        return None

    changed = False

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        changed = True

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        changed = True

    if changed:
        # Reemplaza extensión y evita sobrescribir original
        fixed_path = path.replace(".mp3", "_fixed.wav").replace(".wav", "_fixed.wav")
        torchaudio.save(fixed_path, waveform, sample_rate=TARGET_SR)
        batch["audio"]["path"] = fixed_path
        total_convertidos += 1

    return batch

fixed_dataset = DatasetDict()

for split in dataset:
    print(f"\n🔍 Revisando '{split}'...")
    audio_ds = dataset[split].cast_column("audio", Audio(decode=False))
    fixed_split = audio_ds.map(fix_audio)
    fixed_split = fixed_split.filter(lambda x: x is not None)
    fixed_dataset[split] = fixed_split

print(f"\n💾 Guardando dataset corregido en: {OUTPUT_DATASET}")
fixed_dataset.save_to_disk(OUTPUT_DATASET)

print("\n✅ Proceso finalizado.")
print(f"🎧 Archivos convertidos a 16kHz mono: {total_convertidos}")
print(f"❌ Archivos rotos o no encontrados: {len(errores)}")

if errores:
    print("📄 Lista de archivos con error (primeros 5):")
    for path in errores[:5]:
        print(f" - {path}")
