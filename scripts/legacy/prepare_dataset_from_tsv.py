import os
import torchaudio
from datasets import load_dataset, DatasetDict, Audio

# === CONFIGURACIÓN ===
COMMONVOICE_LANG = "es"
COMMONVOICE_DIR = "data/common_voice"  # carpeta original con los .tsv y .mp3
OUTPUT_DIR = "data/common_voice_subset"  # salida procesada
TARGET_SR = 16000

# === Conversión MP3 -> WAV ===
def convert_to_wav(example):
    mp3_path = os.path.join(COMMONVOICE_DIR, example["path"])
    wav_path = mp3_path.replace(".mp3", ".wav")

    # Si ya está convertido, salta
    if not os.path.exists(wav_path):
        waveform, sr = torchaudio.load(mp3_path)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(wav_path, waveform, TARGET_SR)

    # Devuelve nueva ruta .wav
    example["path"] = wav_path
    return example

# === Cargar y procesar dataset ===
print("📥 Cargando dataset Common Voice...")
dataset = load_dataset("common_voice", COMMONVOICE_LANG, data_dir=COMMONVOICE_DIR)

print("🔄 Convirtiendo audios a 16 kHz mono...")
for split in dataset.keys():
    dataset[split] = dataset[split].map(convert_to_wav)

# === Mantener solo las columnas necesarias ===
columns_to_keep = ["path", "sentence"]
dataset = DatasetDict({
    split: dataset[split].remove_columns(
        [col for col in dataset[split].column_names if col not in columns_to_keep]
    ) for split in dataset
})

# === Cargar como tipo de audio y guardar ===
print("🔗 Cargando audios como tipo 'Audio'...")
dataset = dataset.cast_column("path", Audio(sampling_rate=TARGET_SR))

print(f"💾 Guardando en: {OUTPUT_DIR}")
dataset.save_to_disk(OUTPUT_DIR)

print("✅ Dataset preparado con éxito.")
