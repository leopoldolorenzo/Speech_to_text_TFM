import os
import sys
import torch
import torchaudio
import numpy as np
import json
import requests
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from spellchecker import SpellChecker

# === Configuración modelos ASR ===
ASR_MODELS = {
    "BASE": {
        "path": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        "vocab": "data/vocab/vocab.txt",
        "json_vocab": False
    },
   
    "FT05": {
        "path": "models/fine_tune_05",
        "vocab": "models/fine_tune_05/vocab.json",
        "json_vocab": True
    }
}

# === Configuración modelos de lenguaje ===
LMS = {
    "LM_BASE": "models/lm_base/modelo_limpio.bin",
    "LM_FT": "models/lm_finetune/modelo_finetune.bin",
    "LM_FT_V02": "models/lm_finetuneV02/modelo_finetuneV02.bin",
    "LM_FT_V05": "models/lm_finetuneV05/modelo_finetune05.bin",
    "LM_FT_V06": "models/lm_finetuneV06/modelo_finetune06.bin"
}

AUDIO_INPUT = "data/diarizacion/prueba01.wav"
OUT_DIR = "data/diarizacion/comparacion"
BATCH_SIZE = 8

os.makedirs(OUT_DIR, exist_ok=True)

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

def reconstruir_lexico(texto, spell):
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

def corregir_ortografia(texto, spell):
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

def puntuacion_basica(texto):
    texto = texto.strip()
    if not texto:
        return ""
    if texto[0].isalpha():
        texto = texto[0].upper() + texto[1:]
    if texto[-1] not in [".", "!", "?", "…"]:
        texto += "."
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto

def formatear_tiempo(seg):
    total_seconds = int(seg)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

def main():
    verificar_languagetool()

    print("📂 Cargando audio...")
    waveform, sr = torchaudio.load(AUDIO_INPUT)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    print("🔍 Ejecutando diarización...")
    pipeline = SpeakerDiarization.from_pretrained("models/pyannote_speaker_diarization_ready/config.yaml")
    diarization = pipeline(AUDIO_INPUT)

    segments = []
    current = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * 16000)
        end_sample = int(turn.end * 16000)
        audio_seg = waveform[:, start_sample:end_sample].squeeze(0)

        if current and speaker == current["speaker"] and turn.start - current["end"] < 0.5:
            current["end"] = turn.end
            current["audio"] = torch.cat([current["audio"], audio_seg], dim=-1)
        else:
            if current:
                segments.append(current)
            current = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "audio": audio_seg
            }
    if current:
        segments.append(current)

    spell = SpellChecker(language='es')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for asr_key, asr_info in ASR_MODELS.items():
        print(f"\n{'='*60}")
        print(f"🎙️ Cargando modelo ASR: {asr_key}")
        try:
            processor = Wav2Vec2Processor.from_pretrained(asr_info["path"])
            model = Wav2Vec2ForCTC.from_pretrained(asr_info["path"]).to(device).eval()
        except Exception as e:
            print(f"❌ Error al cargar modelo ASR {asr_key}: {e}")
            continue

        vocab_list = load_vocab(asr_info["vocab"], asr_info["json_vocab"])

        for lm_key, lm_path in LMS.items():
            print(f"\n🔁 Probando con LM: {lm_key}")
            try:
                decoder = build_ctcdecoder(vocab_list, kenlm_model_path=lm_path)
            except Exception as e:
                print(f"⚠️ Error al construir decoder para {asr_key} + {lm_key}: {e}")
                continue

            results_txt = []
            results_json = []

            print("🧠 Ejecutando transcripción por lotes...")
            for i in range(0, len(segments), BATCH_SIZE):
                batch = segments[i:i+BATCH_SIZE]
                waveforms = [s["audio"].numpy().astype(np.float32).tolist() for s in batch]
                inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits.cpu().numpy()

                for seg, logit in zip(batch, logits):
                    raw = decoder.decode(logit).strip().lower()
                    lexico = reconstruir_lexico(raw, spell)
                    ortografia = corregir_ortografia(lexico, spell)
                    gramatica = corregir_gramatica(ortografia)
                    final = puntuacion_basica(gramatica)

                    linea_txt = f"[{formatear_tiempo(seg['start'])} - {formatear_tiempo(seg['end'])}] {seg['speaker']}: {final}"
                    results_txt.append(linea_txt)
                    results_json.append({
                        "speaker": seg["speaker"],
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": final
                    })
                    print(linea_txt)

            # Guardar resultados por combinación
            out_dir_comb = os.path.join(OUT_DIR, f"{asr_key}_{lm_key}")
            os.makedirs(out_dir_comb, exist_ok=True)
            with open(os.path.join(out_dir_comb, "transcripcion.txt"), "w", encoding="utf-8") as ftxt, \
                 open(os.path.join(out_dir_comb, "transcripcion.json"), "w", encoding="utf-8") as fjson:
                ftxt.write("\n".join(results_txt))
                json.dump(results_json, fjson, indent=2, ensure_ascii=False)

            print(f"💾 Resultados guardados en: {out_dir_comb}")

    print("\n✅ Proceso completo para todas las combinaciones.")

if __name__ == "__main__":
    main()
