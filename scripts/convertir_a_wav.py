import os
import torchaudio

# === Configuración ===
INPUT_MP3 = "data/diarizacion/prueba02.mp3"
OUTPUT_DIR = "data/diarizacion"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "prueba02.wav")

# === Crear carpeta si no existe ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Verificar que el archivo existe ===
if not os.path.isfile(INPUT_MP3):
    print(f"❌ El archivo {INPUT_MP3} no existe.")
    exit(1)

# === Forzar el uso de FFmpeg ===
torchaudio.set_audio_backend("ffmpeg")

# === Cargar el mp3 ===
print(f"🎧 Cargando {INPUT_MP3}...")
waveform, sr = torchaudio.load(INPUT_MP3)

# === Convertir a mono si es estéreo ===
if waveform.shape[0] > 1:
    print("🔄 Convirtiendo a mono...")
    waveform = waveform.mean(dim=0, keepdim=True)

# === Resamplear a 16kHz si es necesario ===
if sr != 16000:
    print(f"🔄 Resampleando de {sr} Hz a 16000 Hz...")
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    sr = 16000

# === Guardar como .wav ===
print(f"💾 Guardando {OUTPUT_WAV}...")
torchaudio.save(OUTPUT_WAV, waveform, sample_rate=sr)

print("✅ Conversión completa.")
