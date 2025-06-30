import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pyannote.audio.pipelines import SpeakerDiarization
from spellchecker import SpellChecker

# === Configuración fija ===
ASR_PATH = "models/asr_FT08"
VOCAB_PATH = "models/asr_FT08/vocab.json"
LM_PATH = "models/lm_v09/modelo_lm_v09.binary"
DIARIZATION_CONFIG = "models/pyannote_speaker_diarization_ready/config.yaml"
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_WIDTH = 50

# === Verificación de entrada
if not AUDIO_FILE:
    print("❌ Uso: python transcribir_diarizacion_gpu.py <archivo_audio.wav>")
    sys.exit(1)

# === Normalizador
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Formateo de tiempo
def fmt_time(segundos):
    return str(timedelta(seconds=int(segundos)))

# === Cargar vocabulario
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab_list = [""] * len(vocab_json)
    for token, idx in vocab_json.items():
        vocab_list[idx] = " " if token == "|" else token
    return vocab_list

# === Transcripción individual de segmento
def transcribe(audio_seg, processor, model, decoder):
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(inputs).logits[0].cpu().numpy()
    return normalize(decoder.decode(logits, beam_width=BEAM_WIDTH))

# === Cargar modelos y decodificador
print("🔧 Cargando modelo ASR FT08 y LM_V09 (Optimizado)...")
vocab_list = load_vocab(VOCAB_PATH)
processor = Wav2Vec2Processor.from_pretrained(ASR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(ASR_PATH).to(DEVICE).eval()
decoder = build_ctcdecoder(vocab_list, kenlm_model_path=LM_PATH, alpha=0.5, beta=0.0)

# === Cargar pipeline de diarización offline
print("👥 Cargando modelo de diarización (modo offline)...")
pipeline = SpeakerDiarization.from_pretrained(DIARIZATION_CONFIG)

# === Procesar audio
print(f"📂 Procesando archivo: {AUDIO_FILE}")
audio = AudioSegment.from_wav(AUDIO_FILE)
segments = pipeline(AUDIO_FILE)

# === Transcripción por segmento
results = []
print("✂️ Transcribiendo por segmentos…")

for turn in segments.itertracks(yield_label=True):
    segment, _, speaker = turn
    start_ms = int(segment.start * 1000)
    end_ms = int(segment.end * 1000)
    clip = audio[start_ms:end_ms]

    try:
        text = transcribe(clip, processor, model, decoder)
        print(f"[{fmt_time(segment.start)} - {fmt_time(segment.end)}] {speaker}: {text}")
        results.append({
            "archivo": os.path.basename(AUDIO_FILE),
            "inicio": fmt_time(segment.start),
            "fin": fmt_time(segment.end),
            "speaker": speaker,
            "texto": text
        })
    except Exception as e:
        print(f"❌ Error en segmento [{start_ms}-{end_ms}]: {e}")

# === Corrección ortográfica con SpellChecker
print("📝 Aplicando corrección ortográfica (SpellChecker)...")
spell = SpellChecker(language='es')

for r in results:
    palabras = r["texto"].split()
    corregidas = []

    for p in palabras:
        if spell.unknown([p]):
            sugerida = spell.correction(p)
            corregidas.append(sugerida if sugerida else p)
        else:
            corregidas.append(p)

    r["texto_spell"] = " ".join(corregidas)

# === Guardado de resultados
output_dir = Path("data/diarizacion/comparacion/FT08_LM_V09_Optimizado")
output_dir.mkdir(parents=True, exist_ok=True)

txt_path = output_dir / "transcripcion.txt"
json_path = output_dir / "transcripcion.json"
txt_spell_path = output_dir / "transcripcion_spellcheck.txt"

# Original
with open(txt_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"[{r['inicio']} - {r['fin']}] {r['speaker']}: {r['texto']}\n")

# JSON estructurado
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# Ortografía corregida
with open(txt_spell_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"[{r['inicio']} - {r['fin']}] {r['speaker']}: {r['texto_spell']}\n")

print(f"\n✅ Guardado completado:")
print(f" - {txt_path}")
print(f" - {json_path}")
print(f" - {txt_spell_path}")
