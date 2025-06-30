import os
import json
import csv
import re
from pathlib import Path
import torch
import torchaudio
import evaluate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

# === Configuración ===
ASR_PATH = "models/asr_FT08"
VOCAB_PATH = "models/asr_FT08/vocab.json"
LM_PATH = "models/lm_v09/modelo_lm_v09.binary"
EVAL_DIR = Path("data/eval_confiable_100/wav")
TXT_DIR = Path("data/eval_confiable_100/txt")
OUT_CSV = Path("results/comparacion_todos_modelos_y_lms_mas_optimo2.csv")

# === Métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# === Normalizador
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Cargar vocabulario
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_list = [""] * len(vocab_json)
    for token, idx in vocab_json.items():
        vocab_list[idx] = " " if token == "|" else token
    return vocab_list

# === Transcripción
def transcribe(audio_path, model, processor, decoder, beam_width=50):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    return decoder.decode(logits, beam_width=beam_width).lower()

# === Inicializar modelo y decoder
print("🔧 Cargando modelo ASR y LM …")
vocab_list = load_vocab(VOCAB_PATH)
processor = Wav2Vec2Processor.from_pretrained(ASR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(ASR_PATH)

# Decoders: uno estándar y otro optimizado
decoder_std = build_ctcdecoder(vocab_list, kenlm_model_path=LM_PATH)
decoder_opt = build_ctcdecoder(vocab_list, kenlm_model_path=LM_PATH, alpha=0.5, beta=0.0)

# === Evaluar ambas variantes
nuevos_resultados = []
for tag, decoder, beam in [
    ("LM_V09_Normalizado", decoder_std, 100),
    ("LM_V09_Nor__bw50", decoder_std, 50),
    ("LM_V09_Optimizado", decoder_opt, 50)
]:
    print(f"\n🔍 Evaluando {tag} …")
    for file in sorted(EVAL_DIR.glob("*.wav")):
        txt_path = file.with_suffix(".txt").as_posix().replace("/wav/", "/txt/")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            referencia = normalize(f.read())

        try:
            prediccion = normalize(transcribe(file, model, processor, decoder, beam))
            w = wer.compute(predictions=[prediccion], references=[referencia])
            c = cer.compute(predictions=[prediccion], references=[referencia])
            print(f"🔤 {file.name} → WER {w:.3f} | CER {c:.3f}")
            nuevos_resultados.append([file.name, "FT08", tag, round(w, 4), round(c, 4), prediccion, referencia])
        except Exception as e:
            print(f"❌ Error en {file.name}: {e}")

# === Añadir al CSV existente
print("\n💾 Añadiendo resultados normalizados al CSV final …")
with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(nuevos_resultados)

print("✅ Proceso completado.")
