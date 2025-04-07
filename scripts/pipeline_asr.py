# scripts/pipeline_asr.py (versión sin LM)

import sys
import os
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Ruta al modelo (base o fine-tuned) ===
model_name = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL_DIR = f"models/{model_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()

def transcribe_audio(audio_path: str) -> str:
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python pipeline_asr.py <modelo> <ruta_audio.wav>")
        sys.exit(1)

    _, model_name, audio_path = sys.argv
    transcription = transcribe_audio(audio_path)
    print("\n📝 Transcripción final:")
    print(transcription)
    print("\n✅ Transcripción completada.")
    print("🔊 Audio procesado:", audio_path)
    print("📦 Modelo utilizado:", model_name)
    print("🔤 Transcripción:", transcription)
    print("🔄 Proceso de transcripción finalizado.")
