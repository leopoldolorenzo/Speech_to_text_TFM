# comparar_3_modelos_con_lm.py

import os
import torch
import torchaudio
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import evaluate
import json

# === Rutas ===
EVAL_DIR = "data/eval"

MODELS = {
    "BASE": {
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        "vocab": "data/vocab/vocab.txt",
        "lm": "data/lm/modelo_limpio.bin",
        "json_vocab": False
    },
    "FT01": {
        "model": "models/fine_tune_01",
        "vocab": "models/fine_tune_01/vocab.json",
        "lm": "data/lm/modelo_finetune.bin",
        "json_vocab": True
    },
    "FT02": {
        "model": "models/fine_tune_02",
        "vocab": "models/fine_tune_01/vocab.json",  # usa el mismo tokenizer
        "lm": "data/lm/modelo_finetune.bin",
        "json_vocab": True
    }
}

# === Métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# === Funciones auxiliares
def load_vocab(path, json_mode):
    if json_mode:
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_list = [""] * len(vocab)
        for token, idx in vocab.items():
            vocab_list[idx] = " " if token == "|" else token
        return vocab_list
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

def transcribe(path, model, processor, decoder):
    speech, sr = torchaudio.load(path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    return decoder.decode(logits).lower()

# === Cargar todos los modelos y decoders
loaded = {}
for key, info in MODELS.items():
    print(f"\n📦 Cargando modelo {key}...")
    processor = Wav2Vec2Processor.from_pretrained(info["model"])
    model = Wav2Vec2ForCTC.from_pretrained(info["model"])
    vocab_list = load_vocab(info["vocab"], info["json_vocab"])
    decoder = build_ctcdecoder(vocab_list, kenlm_model_path=info["lm"])
    loaded[key] = {"model": model, "processor": processor, "decoder": decoder}

# === Comparación
print("\n📊 Comparando los 3 modelos con su LM:\n")

for file in sorted(os.listdir(EVAL_DIR)):
    if file.endswith(".wav"):
        wav_path = os.path.join(EVAL_DIR, file)
        txt_path = wav_path.replace(".wav", ".txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            ref = f.read().strip().lower()

        print(f"🎧 Archivo: {file}")
        print(f"REF : {ref}")

        for key in ["BASE", "FT01", "FT02"]:
            pred = transcribe(wav_path, loaded[key]["model"], loaded[key]["processor"], loaded[key]["decoder"])
            w = wer.compute(predictions=[pred], references=[ref])
            c = cer.compute(predictions=[pred], references=[ref])
            print(f"{key:<5}: {pred}")
            print(f"     WER: {w:.2f} | CER: {c:.2f}")
        print("-" * 90)
