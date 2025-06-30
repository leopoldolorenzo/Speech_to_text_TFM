import os
import sys
import torch
import torchaudio
import numpy as np
import json
import requests
from datetime import timedelta
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pyannote.audio import Pipeline
from spellchecker import SpellChecker

# ======================
# CONFIGURACIÓN GENERAL
# ======================
ASR_MODEL = "models/fine_tune_05"
TOKENIZER_PATH = "models/fine_tune_05/vocab.json"
LM_BIN_PATH = "models/lm_finetuneV05/modelo_finetune05.bin"

AUDIO_INPUT = "data/diarizacion/prueba01.wav"
OUT_TXT = "data/diarizacion/diarizacion_transcripcion.txt"
OUT_JSON = "data/diarizacion/diarizacion_transcripcion.json"
BATCH_SIZE = 8

os.makedirs("data/diarizacion", exist_ok=True)

# ===================================
# VERIFICAR CONEXIÓN A LANGUAGETOOL
# ===================================
def verificar_languagetool():
    try:
        r = requests.post("http://localhost:8081/v2/check",
                          data={"text": "Hola mundo", "language": "es"},
                          timeout=5)
        if r.status_code == 200 and "matches" in r.json():
            print("✅ LanguageTool está activo y responde correctamente.")
        else:
            print(f"⚠️ Respuesta inesperada del servidor: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print("❌ No se pudo conectar con LanguageTool en localhost:8081")
        print("   Ejecutá esto en otra terminal:")
        print("   java -cp '*' org.languagetool.server.HTTPServer --port 8081")
        print(f"   Detalles: {e}")
        sys.exit(1)

verificar_languagetool()

# ======================
# CARGA DE MODELO ASR
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🎙️ Cargando modelo ASR...")
processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).to(device).eval()

# ======================
# DECODER CON KENLM
# ======================
print("📖 Cargando vocabulario y modelo de lenguaje...")
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)
vocab_list = [k.replace(" ", "|") if k == " " else k for k, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
decoder = build_ctcdecoder(vocab_list, kenlm_model_path=LM_BIN_PATH)

# ======================
# CARGAR PIPELINE DIARIZACIÓN OFFLINE
# ======================
print("👥 Cargando modelo de diarización (modo offline)...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    cache_dir="models/pyannote_speaker_diarization"
)


# ======================
# SPELLCHECKER
# ======================
spell = SpellChecker(language='es')

# ======================
# FUNCIONES DE CORRECCIÓN
# ======================
def reconstruir_lexico(texto):
    palabras = texto.split()
    reconstruidas = []
    buffer = ""
    i = 0
    while i < len(palabras):
        palabra = palabras[i]
        if spell.known([palabra]):
            if buffer:
                reconstruidas.append(buffer)
                buffer = ""
            reconstruidas.append(palabra)
        else:
            buffer += palabra
            j = i + 1
            while j < len(palabras) and not spell.known([buffer]) and len(buffer) < 20:
                buffer += palabras[j]
                j += 1
            if spell.known([buffer]):
                reconstruidas.append(buffer)
                buffer = ""
                i = j - 1
            else:
                reconstruidas.append(palabra)
                buffer = ""
        i += 1
    if buffer:
        reconstruidas.append(buffer)
    return " ".join(reconstruidas)

def corregir_ortografia(texto):
    palabras = texto.split()
    corregidas = []
    for palabra in palabras:
        if palabra.isalpha() and palabra not in spell:
            sugerencia = spell.correction(palabra)
            corregidas.append(sugerencia if sugerencia else palabra)
        else:
            corregidas.append(palabra)
    return " ".join(corregidas)

def corregir_gramatica(texto):
    try:
        r = requests.post(
            'http://localhost:8081/v2/check',
            data={'text': texto, 'language': 'es'},
            timeout=10
        )
        matches = r.json().get('matches', [])
        for match in reversed(matches):
            offset = match['offset']
            length = match['length']
            replacement = match['replacements'][0]['value'] if match['replacements'] else ''
            texto = texto[:offset] + replacement + texto[offset + length:]
        return texto
    except Exception as e:
        print(f"⚠️ Error al corregir con LanguageTool: {e}")
        return texto

def formatear_tiempo(seg):
    total_seconds = int(seg)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

# ======================
# CARGAR AUDIO
# ======================
print("📂 Cargando audio...")
waveform, sr = torchaudio.load(AUDIO_INPUT)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

# ======================
# DIARIZACIÓN
# ======================
print("🔍 Ejecutando diarización...")
diarization = pipeline(AUDIO_INPUT)

# ======================
# SEGMENTACIÓN
# ======================
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start, end = int(turn.start * 16000), int(turn.end * 16000)
    audio_seg = waveform[:, start:end].squeeze(0)
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker,
        "audio": audio_seg
    })

# ======================
# TRANSCRIPCIÓN POR LOTES
# ======================
print("🧠 Ejecutando transcripción por lotes...")
results_txt = []
results_json = []

for i in range(0, len(segments), BATCH_SIZE):
    batch = segments[i:i+BATCH_SIZE]
    waveforms = [s["audio"].numpy().astype(np.float32).tolist() for s in batch]
    inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()

    for seg, logit in zip(batch, logits):
        raw = decoder.decode(logit).strip().lower()
        lexico = reconstruir_lexico(raw)
        ortografia = corregir_ortografia(lexico)
        final = corregir_gramatica(ortografia)

        linea_txt = f"[{formatear_tiempo(seg['start'])} - {formatear_tiempo(seg['end'])}] {seg['speaker']}: {final}"
        results_txt.append(linea_txt)
        results_json.append({
            "speaker": seg["speaker"],
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": final
        })
        print(linea_txt)

# ======================
# GUARDAR RESULTADOS
# ======================
print("\n💾 Guardando resultados...")
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(results_txt))

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

print(f"\n✅ Transcripción guardada en:\n - {OUT_TXT}\n - {OUT_JSON}")
