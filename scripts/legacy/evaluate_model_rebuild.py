# scripts/evaluate_model_rebuild.py — evaluación de modelo rebuild vs base

import os
import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from jiwer import wer

# === Configuración
AUDIO_DIR = "audios/convertidos"
MODEL_DIRS = {
    "base": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "rebuild": "models/spanish_w2v2_rebuild"
}
TARGETS_FILE = "data/val_dataset.tsv"

# === Cargar pares audio-transcripción
with open(TARGETS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]  # saltar encabezado

examples = []
for line in lines:
    logits_path, transcript = line.strip().split("\t")
    audio_name = os.path.basename(logits_path).replace(".npy", ".wav")
    audio_path = os.path.join(AUDIO_DIR, audio_name)
    if os.path.exists(audio_path):
        examples.append((audio_path, transcript))

# === Función de evaluación
def transcribe_and_evaluate(model_path, name):
    print(f"\n🎤 Evaluando modelo: {name}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()

    predictions = []
    references = []

    for audio_path, ref_text in examples:
        speech, _ = torchaudio.load(audio_path)
        speech = speech.squeeze().numpy()

        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0].lower()

        references.append(ref_text.lower())
        predictions.append(transcription)

        print(f"\n🔊 {os.path.basename(audio_path)}")
        print(f"🎯 Real     : {ref_text}")
        print(f"🤖 Predicho: {transcription}")

    error = wer(references, predictions)
    print(f"\n📊 WER ({name}): {error:.4f}")

# === Ejecutar evaluación para ambos modelos
import torchaudio
for name, model_dir in MODEL_DIRS.items():
    transcribe_and_evaluate(model_dir, name)
