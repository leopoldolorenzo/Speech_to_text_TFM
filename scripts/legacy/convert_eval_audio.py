import torchaudio
import os

EVAL_DIR = "data/eval"
TARGET_SR = 16000

def convert_wav(input_path, output_path):
    waveform, sr = torchaudio.load(input_path)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(output_path, waveform, sample_rate=TARGET_SR)
    print(f"[✅] Convertido: {output_path} | SR: {TARGET_SR}, Mono")

# Archivos a convertir
files = ["prueba01.wav", "prueba02.wav"]

for fname in files:
    in_path = os.path.join(EVAL_DIR, fname)
    out_path = os.path.join(EVAL_DIR, fname.replace(".wav", "_fixed.wav"))
    convert_wav(in_path, out_path)
