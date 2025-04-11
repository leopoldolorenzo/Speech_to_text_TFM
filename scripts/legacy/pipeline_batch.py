import os
import sys
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
VOCAB_PATH = "data/vocab/vocab.txt"
LM_PATH = "data/lm/modelo_limpio.bin"

if len(sys.argv) < 2:
    print("Uso: python pipeline_batch.py carpeta_con_audios/")
    sys.exit(1)

audio_dir = sys.argv[1]
assert os.path.exists(audio_dir), f"Directorio no encontrado: {audio_dir}"

# Preparar salidas
os.makedirs("outputs/logits", exist_ok=True)
os.makedirs("results/batch_transcriptions", exist_ok=True)

# Cargar modelo y tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device).eval()

# Cargar vocabulario y LM
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

decoder = build_ctcdecoder(vocab, kenlm_model_path=LM_PATH)

# Procesar todos los archivos .wav
print(f"🎧 Procesando {len(os.listdir(audio_dir))} archivos en: {audio_dir}\n")

for fname in sorted(os.listdir(audio_dir)):
    if not fname.endswith(".wav"):
        continue

    fpath = os.path.join(audio_dir, fname)
    name = os.path.splitext(fname)[0]

    # Cargar audio
    audio, _ = librosa.load(fpath, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

    # Obtener logits y guardarlos
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0).cpu().numpy()
    logits_path = f"outputs/logits/{name}.npy"
    np.save(logits_path, logits)

    # Decodificar logits
    transcription = decoder.decode(logits)

    # Guardar transcripción
    with open(f"results/batch_transcriptions/{name}.txt", "w", encoding="utf-8") as f:
        f.write(transcription.strip())

    print(f"✅ {fname} → results/batch_transcriptions/{name}.txt")

print("\n📁 Transcripciones guardadas en: results/batch_transcriptions")
print("📁 Logits guardados en: outputs/logits/")
print("✅ Proceso completado.")
# End of script
