import os
import json
import csv
import re
from pathlib import Path
import torch
import torchaudio
import evaluate
import requests
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from spellchecker import SpellChecker

# === Configuración ===
ASR_PATH = "models/asr_FT08"
VOCAB_PATH = "models/asr_FT08/vocab.json"
LM_PATH = "models/lm_v09/modelo_lm_v09.binary"
EVAL_DIR = Path("data/eval_confiable_100/wav")
TXT_DIR = Path("data/eval_confiable_100/txt")
OUT_CSV = Path("results/comparacion_todos_modelos_y_lms_mas_optimo2.csv")
APLICAR_MEJORAS = True  # Cambiar a False si no deseas mejoras lingüísticas

# === Métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# === Inicializar corrector ortográfico
spell = SpellChecker()

# === Normalizador
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Cargar vocabulario
def load_vocab(path, json_mode=True):
    path = Path(path)
    if json_mode:
        vocab_json = json.loads(path.read_text(encoding="utf-8"))
        vocab_list = [""] * len(vocab_json)
        for token, idx in vocab_json.items():
            vocab_list[idx] = " " if token == "|" else token
        return vocab_list
    else:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]

# === Corrección ortográfica
def corregir_ortografia(texto):
    palabras = texto.split()
    corregidas = []
    for p in palabras:
        if p.isalpha():
            cor = spell.correction(p)
            corregidas.append(cor if cor is not None else p)
        else:
            corregidas.append(p)
    return " ".join(corregidas)

# === Corrección gramatical
def corregir_gramatica(texto):
    try:
        r = requests.post(
            "http://localhost:8081/v2/check",
            data={"text": texto, "language": "es"},
            timeout=5
        )
        if r.ok:
            matches = r.json().get("matches", [])
            texto_lista = list(texto)
            for m in reversed(matches):
                offset = m.get("offset", 0)
                length = m.get("length", 0)
                replacements = m.get("replacements", [])
                if replacements:
                    reemplazo = replacements[0].get("value")
                    if reemplazo is not None:
                        texto_lista[offset:offset+length] = list(reemplazo)
            texto = "".join(texto_lista)
    except Exception as e:
        print(f"⚠️ Error en corrección gramatical: {e}")
    return texto

# === Puntuación básica
def puntuacion_basica(texto):
    texto = texto.strip()
    if texto and texto[-1] not in ".!?":
        texto += "."
    if texto:
        texto = texto[0].upper() + texto[1:]
    return texto

# === Transcripción
def transcribe(audio_path, model, processor, decoder, beam_width=50, aplicar_mejoras=False):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    texto = decoder.decode(logits, beam_width=beam_width).lower()
    if aplicar_mejoras:
        texto = corregir_ortografia(texto)
        texto = corregir_gramatica(texto)
        texto = puntuacion_basica(texto)
    return texto

# === Inicializar modelo y decoder
print("🔧 Cargando modelo ASR y LM …")
vocab_list = load_vocab(VOCAB_PATH, json_mode=True)
processor = Wav2Vec2Processor.from_pretrained(ASR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(ASR_PATH)

decoder_std = build_ctcdecoder(vocab_list, kenlm_model_path=LM_PATH)
decoder_opt = build_ctcdecoder(vocab_list, kenlm_model_path=LM_PATH, alpha=1.5, beta=0.5)

# === Evaluación
nuevos_resultados = []
for tag, decoder, beam in [
    ("LM_V09_NORM", decoder_std, 50),
    ("LM_V09_OPT_NORM", decoder_opt, 100)
]:
    print(f"\n🔍 Evaluando {tag} {'(con mejoras)' if APLICAR_MEJORAS else ''} …")
    for file in sorted(EVAL_DIR.glob("*.wav")):
        txt_path = file.with_suffix(".txt").as_posix().replace("/wav/", "/txt/")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            referencia = normalize(f.read())

        try:
            pred_cruda = transcribe(file, model, processor, decoder, beam, aplicar_mejoras=APLICAR_MEJORAS)
            pred_norm = normalize(pred_cruda)
            w = wer.compute(predictions=[pred_norm], references=[referencia])
            c = cer.compute(predictions=[pred_norm], references=[referencia])
            print(f"🔤 {file.name} → WER {w:.3f} | CER {c:.3f}")
            nuevos_resultados.append([file.name, "FT08", tag, round(w, 4), round(c, 4), pred_cruda, referencia])
        except Exception as e:
            print(f"❌ Error en {file.name}: {e}")

# === Guardar resultados
print("\n💾 Añadiendo resultados al CSV final …")
with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(nuevos_resultados)

print("✅ Proceso completado.")
