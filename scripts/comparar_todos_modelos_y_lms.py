# comparar_todos_modelos_y_lms.py

import os
import torch
import torchaudio
import evaluate
import json
import csv
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

# === Rutas ===
EVAL_DIR = "data/eval_confiable/wav"
OUT_CSV = "results/comparacion_todos_modelos_y_lms.csv"

# === Modelos ASR ===
# === Modelos ASR seleccionados ===
ASR_MODELS = {
    "BASE": {
        "path": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        "vocab": "data/vocab/vocab.txt",
        "json_vocab": False
    },
    "FT03": {
        "path": "models/fine_tune_03",
        "vocab": "models/fine_tune_03/vocab.json",
        "json_vocab": True
    },
    "FT05": {
        "path": "models/fine_tune_05",
        "vocab": "models/fine_tune_05/vocab.json",
        "json_vocab": True
    },
    "FT08": {
        "path": "models/fine_tune_08",
        "vocab": "models/fine_tune_08/vocab.json",
        "json_vocab": True
    }
}


# === Modelos de lenguaje ===
LMS = {
    "LM_BASE": "models/lm_base/modelo_limpio.bin",      
    "LM_FT": "models/lm_finetune/modelo_finetune.bin",
    "LM_FT_V02": "models/lm_finetuneV02/modelo_finetuneV02.bin",    
    "LM_FT_V05": "models/lm_finetuneV05/modelo_finetune05.bin",
    "LM_FT_V06": "models/lm_finetuneV06/modelo_finetune06.bin",

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

def transcribe(audio_path, model, processor, decoder):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    return decoder.decode(logits).lower()

# === Comparación
results = [["archivo", "modelo_asr", "modelo_lm", "WER", "CER", "prediccion", "referencia"]]

for asr_key, asr_info in ASR_MODELS.items():
    print(f"\n{'='*60}")
    print(f"📦 Cargando modelo ASR: {asr_key}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(asr_info["path"])
        model = Wav2Vec2ForCTC.from_pretrained(asr_info["path"])
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

        for file in sorted(os.listdir(EVAL_DIR)):
            if not file.endswith(".wav"):
                continue

            wav_path = os.path.join(EVAL_DIR, file)
            txt_path = wav_path.replace(".wav", ".txt").replace("wav", "txt")
            if not os.path.exists(txt_path):
                print(f"⚠️ Falta .txt para {file}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                referencia = f.read().strip().lower()

            try:
                prediccion = transcribe(wav_path, model, processor, decoder)
                w = wer.compute(predictions=[prediccion], references=[referencia])
                c = cer.compute(predictions=[prediccion], references=[referencia])
                print(f"🎧 {file} | {asr_key}+{lm_key} → WER: {w:.2f}, CER: {c:.2f}")
                results.append([file, asr_key, lm_key, round(w, 4), round(c, 4), prediccion, referencia])
            except Exception as e:
                print(f"❌ Error al transcribir {file} con {asr_key}+{lm_key}: {e}")

# === Guardar CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(results)

print(f"\n✅ Comparación finalizada. Resultados guardados en: {OUT_CSV}")


