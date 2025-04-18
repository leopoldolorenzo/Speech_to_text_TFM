import os
import torch
import torchaudio
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import evaluate
import json

# === Paths ===
EVAL_DIR = "data/eval"
KENLM_BASE = "data/lm/modelo_limpio.bin"
KENLM_FT = "data/lm/modelo_finetune.bin"
VOCAB_BASE = "data/vocab/vocab.txt"
VOCAB_FT = "models/fine_tune_01/vocab.json"
MODEL_BASE = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
MODEL_FT = "models/fine_tune_01"

# === Carga de métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# === Utilidades ===
def load_vocab_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_list = [""] * len(vocab)
    for token, idx in vocab.items():
        vocab_list[idx] = " " if token == "|" else token
    return vocab_list

def load_vocab_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# === Decoders
vocab_base = load_vocab_from_txt(VOCAB_BASE)
decoder_base = build_ctcdecoder(vocab_base, kenlm_model_path=KENLM_BASE)

vocab_ft = load_vocab_from_json(VOCAB_FT)
decoder_ft = build_ctcdecoder(vocab_ft, kenlm_model_path=KENLM_FT)

# === Modelos
print("📦 Cargando modelo base...")
model_base = Wav2Vec2ForCTC.from_pretrained(MODEL_BASE)
processor_base = Wav2Vec2Processor.from_pretrained(MODEL_BASE)

print("📦 Cargando modelo fine-tuneado...")
model_ft = Wav2Vec2ForCTC.from_pretrained(MODEL_FT)
processor_ft = Wav2Vec2Processor.from_pretrained(MODEL_FT)

# === Transcripción con LM
def transcribe(path, model, processor, decoder):
    speech, sr = torchaudio.load(path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    return decoder.decode(logits).lower()

# === Evaluación
print("\n📊 Comparando modelos (ambos con su LM respectivo):\n")

for file in sorted(os.listdir(EVAL_DIR)):
    if file.endswith(".wav"):
        wav_path = os.path.join(EVAL_DIR, file)
        txt_path = wav_path.replace(".wav", ".txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            ref = f.read().strip().lower()

        base_pred = transcribe(wav_path, model_base, processor_base, decoder_base)
        ft_pred = transcribe(wav_path, model_ft, processor_ft, decoder_ft)

        wer_base = wer.compute(predictions=[base_pred], references=[ref])
        wer_ft = wer.compute(predictions=[ft_pred], references=[ref])
        cer_base = cer.compute(predictions=[base_pred], references=[ref])
        cer_ft = cer.compute(predictions=[ft_pred], references=[ref])

        print(f"🎧 Archivo: {file}")
        print(f"REF : {ref}")
        print(f"BASE: {base_pred}")
        print(f"FT  : {ft_pred}")
        print(f"WER Base: {wer_base:.2f} | FT: {wer_ft:.2f}")
        print(f"CER Base: {cer_base:.2f} | FT: {cer_ft:.2f}")
        print("-" * 80)
