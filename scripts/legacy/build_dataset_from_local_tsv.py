import os
import torchaudio
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

# === CONFIGURACIÓN ===
COMMONVOICE_DIR = "data/common_voice"
OUTPUT_DIR = "data/common_voice_subset"
TARGET_SR = 16000
MAX_TRAIN = 1000
MAX_TEST = 200

# === Conversión de audio ===
def convert_audio(row):
    mp3_path = os.path.join(COMMONVOICE_DIR, "clips", row["path"])
    wav_path = mp3_path.replace(".mp3", ".wav")

    if not os.path.exists(wav_path):
        waveform, sr = torchaudio.load(mp3_path)
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(wav_path, waveform, sample_rate=TARGET_SR)

    return {"audio": wav_path, "sentence": row["sentence"]}

# === Proceso para train/test ===
def process_split(split_name, max_items):
    print(f"🔄 Procesando {split_name} (máx {max_items})...")
    tsv_path = os.path.join(COMMONVOICE_DIR, f"{split_name}.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[["path", "sentence"]].dropna().head(max_items)
    data = [convert_audio(row) for row in df.to_dict(orient="records")]
    return Dataset.from_list(data).cast_column("audio", Audio(sampling_rate=TARGET_SR))

# === Construcción y guardado ===
print("📦 Cargando y procesando dataset Common Voice local...")

dataset = DatasetDict({
    "train": process_split("train", MAX_TRAIN),
    "test": process_split("test", MAX_TEST),
})

print(f"💾 Guardando en: {OUTPUT_DIR}")
dataset.save_to_disk(OUTPUT_DIR)
print("✅ Dataset reducido y guardado con éxito.")
