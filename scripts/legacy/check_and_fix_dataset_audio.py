# scripts/check_and_fix_dataset_audio.py
import os
import torchaudio
from datasets import load_from_disk, DatasetDict, Audio

INPUT_DATASET = "data/common_voice_subset"
OUTPUT_DATASET = "data/common_voice_subset_fixed"
TARGET_SR = 16000

print(f"[📦] Cargando dataset desde: {INPUT_DATASET}")
dataset = load_from_disk(INPUT_DATASET)

def fix_audio(batch):
    path = batch["audio"]["path"]
    waveform, sr = torchaudio.load(path)

    changed = False
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        changed = True

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        changed = True

    if changed:
        # Guardar audio temporal convertido
        fixed_path = path.replace(".wav", "_fixed.wav")
        torchaudio.save(fixed_path, waveform, sample_rate=TARGET_SR)
        batch["audio"]["path"] = fixed_path

    return batch

fixed_dataset = DatasetDict()

for split in dataset:
    print(f"\n🔍 Revisando '{split}'...")
    audio_ds = dataset[split].cast_column("audio", Audio(decode=False))
    fixed_split = audio_ds.map(fix_audio)
    fixed_dataset[split] = fixed_split

print(f"\n💾 Guardando dataset corregido en: {OUTPUT_DATASET}")
fixed_dataset.save_to_disk(OUTPUT_DATASET)
print("[✅] Proceso finalizado.")
