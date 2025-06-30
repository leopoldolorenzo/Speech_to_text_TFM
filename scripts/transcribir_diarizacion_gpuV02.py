#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcripción + diarización + spell-check + LanguageTool local (HTTP)

Uso:
    python transcribir_diarizacion_gpuV03.py <archivo.wav> [PUERTO_LT]
Ejemplo:
    python transcribir_diarizacion_gpuV03.py data/diarizacion/prueba01.wav 8081
"""

import os, sys, re, json, requests, numpy as np
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment

import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pyannote.audio.pipelines import SpeakerDiarization
from spellchecker import SpellChecker
import language_tool_python                       # ←  AHORA sí se importa

# ─────────────────── Configuración básica ────────────────────
ASR_PATH           = "models/asr_FT08"
VOCAB_PATH         = "models/asr_FT08/vocab.json"
LM_PATH            = "models/lm_v09/modelo_lm_v09.binary"
DIARIZATION_CONFIG = "models/pyannote_speaker_diarization_ready/config.yaml"
BEAM_WIDTH         = 50

# ───────────────────── Utilidades varias ─────────────────────
def normalize(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^\w\sáéíóúñü]", "", txt)
    return re.sub(r"\s+", " ", txt).strip()

def t(seg):  # formatea segundos → hh:mm:ss
    return str(timedelta(seconds=int(seg)))

def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    vocab = [""] * len(j)
    for tok, idx in j.items():
        vocab[idx] = " " if tok == "|" else tok
    return vocab

def prepare_for_lt(txt: str) -> str:
    txt = txt.strip()
    if txt and txt[0].islower():
        txt = txt[0].upper() + txt[1:]
    if txt and txt[-1] not in ".!?":
        txt += "."
    return txt

def transcribe(audio_seg, processor, model, decoder):
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
    inp = processor(samples, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(inp).logits[0].cpu().numpy()
    return normalize(decoder.decode(logits, beam_width=BEAM_WIDTH))

# ───────────── LanguageTool vía HTTP (handler propio) ──────────────
def lt_correct(text, *, host="localhost", port=8081, lang="es", timeout=15):
    """Devuelve el texto corregido usando el servidor LT local."""
    url = f"http://{host}:{port}/v2/check"
    rsp = requests.post(url,
                        data={"language": lang, "text": text},
                        timeout=timeout)
    rsp.raise_for_status()
    matches = rsp.json().get("matches", [])

    # Aplicamos reemplazos en orden inverso para no desplazar offsets
    corrected = list(text)
    for m in sorted(matches, key=lambda m: m["offset"], reverse=True):
        rep = m["replacements"][0]["value"] if m["replacements"] else None
        if rep:
            off, length = m["offset"], m["length"]
            corrected[off:off+length] = rep
    return "".join(corrected)

# ──────────────────────────── Main ────────────────────────────
if len(sys.argv) < 2:
    print("❌ Uso: python transcribir_diarizacion_gpuV03.py <audio.wav> [PUERTO_LT]")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]
LT_PORT    = int(sys.argv[2]) if len(sys.argv) > 2 else 8081
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔧  Cargando modelos ASR + LM …")
vocab      = load_vocab(VOCAB_PATH)
processor  = Wav2Vec2Processor.from_pretrained(ASR_PATH)
model      = Wav2Vec2ForCTC.from_pretrained(ASR_PATH).to(DEVICE).eval()
decoder    = build_ctcdecoder(vocab, kenlm_model_path=LM_PATH, alpha=0.5, beta=0.0)

print("👥  Cargando diarización …")
pipeline   = SpeakerDiarization.from_pretrained(DIARIZATION_CONFIG)

print(f"📂  Procesando: {AUDIO_FILE}")
wav        = AudioSegment.from_wav(AUDIO_FILE)
segments   = pipeline(AUDIO_FILE)

results = []
print("✂️  Transcribiendo segmentos …")
for segment, _, spk in segments.itertracks(yield_label=True):
    clip = wav[int(segment.start*1000): int(segment.end*1000)]
    try:
        txt = transcribe(clip, processor, model, decoder)
        print(f"[{t(segment.start)} - {t(segment.end)}] {spk}: {txt}")
        results.append({"archivo": os.path.basename(AUDIO_FILE),
                        "inicio": t(segment.start),
                        "fin":    t(segment.end),
                        "speaker": spk,
                        "texto":   txt})
    except Exception as e:
        print(f"   ⚠️  Error {segment}: {e}")

# ───────────── SpellChecker ─────────────
print("📝  Spell-check …")
spell = SpellChecker(language="es")
for r in results:
    corr = [spell.correction(p) if spell.unknown([p]) else p
            for p in r["texto"].split()]
    r["texto_spell"] = " ".join(corr)

# ───────────── LanguageTool ─────────────
print(f"🧠  Grammar check via LT (puerto {LT_PORT}) …")
for r in results:
    base = prepare_for_lt(r["texto_spell"])
    try:
        r["texto_lt"] = lt_correct(base, port=LT_PORT)
    except Exception as err:
        print(f"   ⚠️  No se pudo corregir ‘{base[:30]}…’: {err}")
        r["texto_lt"] = base

# ───────────── Guardar ─────────────
out = Path("data/diarizacion/comparacion/FT08_LM_V09_Optimizado")
out.mkdir(parents=True, exist_ok=True)

def save(key, fname):
    with open(out/fname, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"[{r['inicio']} - {r['fin']}] {r['speaker']}: {r[key]}\n")

save("texto",        "transcripcion.txt")
save("texto_spell",  "transcripcion_spellcheck.txt")
save("texto_lt",     "transcripcion_languagetool.txt")

with open(out/"transcripcion.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅  Guardado en:")
for f in ("transcripcion.txt",
          "transcripcion_spellcheck.txt",
          "transcripcion_languagetool.txt",
          "transcripcion.json"):
    print("   •", out/f)
