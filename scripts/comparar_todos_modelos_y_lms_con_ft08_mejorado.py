# scripts/comparar_todos_modelos_y_lms_con_ft08_mejorado.py

import os
import torch
import torchaudio
import evaluate
import json
import csv
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

# === Rutas ===
EVAL_DIR = Path("data/eval_confiable_100/wav")
OUT_CSV = Path("results/comparacion_todos_modelos_y_lms_mas_optimo.csv")

# === Modelos ASR ===
ASR_MODELS = {
    "BASE": {"path": "models/base", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT01": {"path": "models/asr_FT01", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT02": {"path": "models/asr_FT02", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT03": {"path": "models/asr_FT03", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT05": {"path": "models/asr_FT05", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT08": {"path": "models/asr_FT08", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
}

# === Modelos de lenguaje ===
LMS = {
    "LM_BASE": "models/lm_base/modelo_limpio.binary",
    "LM_V01": "models/lm_v01/modelo_finetune.binary",
    "LM_V02": "models/lm_v02/modelo_finetuneV02.binary",
    "LM_V05": "models/lm_v05/modelo_finetune05.binary",
    "LM_V06": "models/lm_v06/modelo_finetune06.binary",
    "LM_V08": "models/lm_v08/modelo_lm_v08.binary",
    "LM_V09": "models/lm_v09/modelo_lm_v09.binary"
}

# === Métricas ===
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# === Función para cargar vocabulario ===
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

# === Transcripción ===
def transcribe(audio_path, model, processor, decoder, beam_width=50):
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    return decoder.decode(logits, beam_width=beam_width).lower()

# === Evaluación principal ===
results = [["archivo", "modelo_asr", "modelo_lm", "WER", "CER", "prediccion", "referencia"]]

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
        print(f"🔁  {asr_key} + {lm_key}")
        try:
            decoder = build_ctcdecoder(vocab_list, kenlm_model_path=lm_path)
        except Exception as e:
            print(f"⚠️  Error construyendo decoder para {asr_key}+{lm_key}: {e}")
            continue

        for file in sorted(EVAL_DIR.glob("*.wav")):
            txt_path = file.with_suffix(".txt").as_posix().replace("/wav/", "/txt/")
            if not os.path.exists(txt_path):
                print(f"⚠️  Falta .txt para {file.name}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                referencia = f.read().strip().lower()

            try:
                prediccion = transcribe(file, model, processor, decoder)
                w = wer.compute(predictions=[prediccion], references=[referencia])
                c = cer.compute(predictions=[prediccion], references=[referencia])
                print(f"🎧 {file.name} → WER {w:.2f} | CER {c:.2f}")
                results.append([file.name, asr_key, lm_key, round(w, 4), round(c, 4), prediccion, referencia])
            except Exception as e:
                print(f"❌  Error en {file.name} ({asr_key}+{lm_key}): {e}")

# === Evaluación adicional con hiperparámetros óptimos ===
print("\n🌟 Evaluando combinación óptima: FT08 + LM_V09 con hiperparámetros óptimos")

asr_path = ASR_MODELS["FT08"]["path"]
vocab_path = ASR_MODELS["FT08"]["vocab"]
vocab_list = load_vocab(vocab_path, True)
processor = Wav2Vec2Processor.from_pretrained(asr_path)
model = Wav2Vec2ForCTC.from_pretrained(asr_path)
decoder = build_ctcdecoder(vocab_list, kenlm_model_path=LMS["LM_V09"], alpha=0.5, beta=0.0)

for file in sorted(EVAL_DIR.glob("*.wav")):
    txt_path = file.with_suffix(".txt").as_posix().replace("/wav/", "/txt/")
    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        referencia = f.read().strip().lower()

    try:
        prediccion = transcribe(file, model, processor, decoder, beam_width=50)
        w = wer.compute(predictions=[prediccion], references=[referencia])
        c = cer.compute(predictions=[prediccion], references=[referencia])
        print(f"🌟 {file.name} → WER {w:.2f} | CER {c:.2f}")
        results.append([file.name, "FT08", "LM_V09_OPT", round(w, 4), round(c, 4), prediccion, referencia])
    except Exception as e:
        print(f"❌ Error en combinación óptima: {e}")

# === Guardar resultados ===
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(results)

print(f"\n✅ Comparación completa. Resultados guardados en: {OUT_CSV}")
