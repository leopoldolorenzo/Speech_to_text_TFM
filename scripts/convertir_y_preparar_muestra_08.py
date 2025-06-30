#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convertir_y_preparar_muestra_08.py
----------------------------------

• Selecciona N muestras de Common Voice ES sin repetir pistas previas.
• Convierte .mp3 → .wav (mono, 16 kHz) con torchaudio (o sox si falla).
• Normaliza texto: minúsculas, sin puntuación, separador CTC "|".
• Verifica que el vocabulario cargado es idéntico al del modelo base.
• Filtra por duración mínima y máxima.
• Registra descartes y pistas utilizadas.

Estructura de salida:
  data/common_voice_es/es/fine_tune_08/
    ├── clips_wav/
    ├── dataset_100k.tsv
    ├── descartes_oov.log
    └── pistas_usadas_08.txt
"""

import csv
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import torchaudio
from transformers import Wav2Vec2Processor

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100_000, help="Número de muestras a seleccionar (default: 100000)")
parser.add_argument("--lote", type=str, default="fine_tune_08", help="Nombre del lote (carpeta de salida)")
parser.add_argument("--previas", type=str, default="pistas_usadas_previas.txt", help="Archivo con pistas usadas previamente")
args = parser.parse_args()

# ---------- CONFIGURACIÓN ----------
N_MUESTRAS            = args.n
LOTE                  = args.lote
PREVIAS_TXT           = args.previas

CV_ROOT               = Path("data/common_voice_es/es")
INPUT_TSV             = CV_ROOT / "validated.tsv"
INPUT_CLIPS_DIR       = CV_ROOT / "clips"
TOKENIZER_DIR         = Path("models/tokenizer_v2_base41")

OUT_DIR               = CV_ROOT / LOTE
OUT_WAV_DIR           = OUT_DIR / "clips_wav"
OUT_TSV               = OUT_DIR / f"dataset_{N_MUESTRAS//1000}k.tsv"
OUT_PISTAS_USADAS     = OUT_DIR / f"pistas_usadas_{LOTE}.txt"
LOG_OOV               = OUT_DIR / "descartes_oov.log"

MIN_DUR_S, MAX_DUR_S  = 0.25, 14.0  # duración aceptable

# ---------- PREPARACIÓN ----------
OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- VERIFICACIÓN VOCABULARIO ----------
try:
    processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
    vocab = processor.tokenizer.get_vocab()
    local_vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    base_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    base_vocab = base_processor.tokenizer.get_vocab()
    base_vocab_sorted = sorted(base_vocab.items(), key=lambda x: x[1])
except Exception as e:
    sys.exit(f"❌ Error cargando tokenizer: {e}")

if local_vocab_sorted != base_vocab_sorted:
    sys.exit("❌ El vocabulario del tokenizer local NO coincide con el del modelo base original.")

print(f"✅ Vocabulario verificado — {len(local_vocab_sorted)} tokens idénticos al modelo base.")

vocab_chars = set(vocab.keys())

# ---------- CARGA DE PISTAS PREVIAS ----------
pistas_previas = set()
if Path(PREVIAS_TXT).is_file():
    with open(PREVIAS_TXT, encoding="utf-8") as f:
        pistas_previas = {l.strip() for l in f if l.strip()}

# ---------- LIMPIEZA TEXTO ----------
punct_re   = re.compile(r"[¿¡,.;:«»!?\"()]+")
spaces_re  = re.compile(r"\s+")

def norm_text(txt: str) -> str:
    txt = punct_re.sub("", txt.lower())
    txt = spaces_re.sub(" ", txt).strip()
    return txt.replace(" ", "|")

# ---------- FUNCIONES AUXILIARES ----------
def mp3_to_wav(mp3: Path, wav: Path) -> bool:
    try:
        wf, sr = torchaudio.load(mp3)
    except RuntimeError:
        cmd = ["sox", str(mp3), "-c", "1", "-r", "16000", str(wav)]
        return subprocess.call(cmd, stderr=subprocess.DEVNULL) == 0
    wf = torchaudio.functional.resample(wf, sr, 16000)
    wf = wf.mean(dim=0, keepdim=True)
    torchaudio.save(str(wav), wf, 16000)
    return True

def duracion_valida(wav: Path) -> bool:
    info = torchaudio.info(str(wav))
    dur = info.num_frames / info.sample_rate
    return MIN_DUR_S <= dur <= MAX_DUR_S

# ---------- CARGA TSV Y FILTRADO ----------
print(f"📥 Leyendo {INPUT_TSV} …")
with open(INPUT_TSV, encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f, delimiter="\t")
        if r["path"] and r["sentence"] and r["path"] not in pistas_previas
    ]

print(f"👉 Candidatas tras filtrar previas: {len(rows):,}")
random.shuffle(rows)
rows = rows[:N_MUESTRAS]
print(f"🎯 Seleccionadas {len(rows):,} muestras para {LOTE}\n")

# ---------- PROCESAMIENTO PRINCIPAL ----------
descartes = []

with open(OUT_TSV, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out, delimiter="\t")

    for r in tqdm(rows, desc="Procesando"):
        mp3_file = INPUT_CLIPS_DIR / r["path"]
        wav_file = OUT_WAV_DIR / r["path"].replace(".mp3", ".wav")

        if not mp3_to_wav(mp3_file, wav_file):
            print(f"⚠️  Error al convertir {mp3_file}")
            continue
        if not duracion_valida(wav_file):
            wav_file.unlink(missing_ok=True)
            continue

        txt = norm_text(r["sentence"])
        if any(ch not in vocab_chars for ch in txt):
            descartes.append(f"{wav_file}\t{txt}")
            wav_file.unlink(missing_ok=True)
            continue

        writer.writerow([wav_file, txt])

# ---------- GUARDAR PISTAS Y LOGS ----------
with open(OUT_PISTAS_USADAS, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(r["path"] + "\n")

with open(PREVIAS_TXT, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(r["path"] + "\n")

if descartes:
    LOG_OOV.write_text("\n".join(descartes), encoding="utf-8")
    print(f"⚠️  Descartadas {len(descartes):,} líneas OOV o inválidas (→ {LOG_OOV})")

print("\n✅ Conversión completada")
print(f"   TSV final : {OUT_TSV}")
print(f"   WAVs en   : {OUT_WAV_DIR}")
print(f"   Pistas OK : {OUT_PISTAS_USADAS}")
print(f"   Pistas acumuladas en: {PREVIAS_TXT}")
