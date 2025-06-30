import os
import torch
import torchaudio
import evaluate
import json
import csv
import re
import requests
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from spellchecker import SpellChecker

EVAL_DIR = Path("data/eval_confiable/wav")
OUT_CSV = Path("results/comparacion_todos_modelos_y_lms_con_mejoras.csv")

ASR_MODELS = {
    "BASE": {"path": "models/base", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT01": {"path": "models/asr_FT01", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT02": {"path": "models/asr_FT02", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT03": {"path": "models/asr_FT03", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT05": {"path": "models/asr_FT05", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT08": {"path": "models/asr_FT08", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True}
}

LMS = {
    "LM_BASE": "models/lm_base/modelo_limpio.binary",
    "LM_V01": "models/lm_v01/modelo_finetune.binary",
    "LM_V02": "models/lm_v02/modelo_finetuneV02.binary",
    "LM_V05": "models/lm_v05/modelo_finetune05.binary",
    "LM_V06": "models/lm_v06/modelo_finetune06.binary",
    "LM_V08": "models/lm_v08/modelo_lm_v08.binary"
}

wer = evaluate.load("wer")
cer = evaluate.load("cer")
spell = SpellChecker(language='es')

def load_vocab(path, json_mode):
    path = Path(path)
    if json_mode:
        vocab_json = json.loads(path.read_text(encoding="utf-8"))
        vocab_list = [""] * len(vocab_json)
        for token, idx in vocab_json.items():
            vocab_list[idx] = " " if token == "|" else token
        return vocab_list
    else:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]

def corregir_ortografia(texto):
    return " ".join(spell.correction(p) if p.isalpha() else p for p in texto.split())

def corregir_gramatica(texto):
    try:
        r = requests.post("http://localhost:8081/v2/check", data={"text": texto, "language": "es"}, timeout=5)
        if r.ok:
            for m in reversed(r.json().get("matches", [])):
                off, length = m["offset"], m["length"]
                rep = m.get("replacements", [])
                if rep:
                    texto = texto[:off] + rep[0]["value"] + texto[off+length:]
    except:
        pass
    return texto

def puntuacion_basica(texto):
    texto = texto.strip()
    if texto and texto[-1] not in ".!?":
        texto += "."
    if texto:
        texto = texto[0].upper() + texto[1:]
    return texto

def transcribe(audio_path, model, processor, decoder, aplicar_mejoras=False):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    texto = decoder.decode(logits).lower()
    if aplicar_mejoras:
        texto = corregir_ortografia(texto)
        texto = corregir_gramatica(texto)
        texto = puntuacion_basica(texto)
    return texto

results = [["archivo", "modelo_asr", "modelo_lm", "modo", "WER", "CER", "prediccion", "referencia"]]

for asr_key, asr_info in ASR_MODELS.items():
    print(f"\n📦 Cargando ASR: {asr_key}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(asr_info["path"])
        model = Wav2Vec2ForCTC.from_pretrained(asr_info["path"])
    except Exception as e:
        print(f"❌ Error al cargar ASR {asr_key}: {e}")
        continue

    vocab_list = load_vocab(asr_info["vocab"], asr_info["json_vocab"])

    for lm_key, lm_path in LMS.items():
        print(f"🔁 {asr_key} + {lm_key}")
        try:
            decoder = build_ctcdecoder(vocab_list, kenlm_model_path=lm_path)
        except Exception as e:
            print(f"⚠️ Error en {asr_key}+{lm_key}: {e}")
            continue

        for file in sorted(EVAL_DIR.glob("*.wav")):
            txt_path = file.with_suffix(".txt").as_posix().replace("/wav/", "/txt/")
            if not os.path.exists(txt_path):
                print(f"⚠️ Falta .txt para {file.name}")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                referencia = f.read().strip().lower()

            for modo in [False, True]:
                try:
                    pred = transcribe(file, model, processor, decoder, aplicar_mejoras=modo)
                    w = wer.compute(predictions=[pred], references=[referencia])
                    c = cer.compute(predictions=[pred], references=[referencia])
                    label = "MEJORAS" if modo else "BASE"
                    print(f"🎧 {file.name} | {label} → WER {w:.2f} | CER {c:.2f}")
                    results.append([file.name, asr_key, lm_key, label, round(w,4), round(c,4), pred, referencia])
                except Exception as e:
                    print(f"❌ Error en {file.name} ({asr_key}+{lm_key}, {label}): {e}")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(results)

print(f"\n✅ Resultados guardados en: {OUT_CSV}")
