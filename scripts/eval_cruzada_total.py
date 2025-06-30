import os
import sys
import json
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

# === FILTRAR WARNINGS de pyctcdecode
class FilterStderr:
    def write(self, msg):
        if "Unigrams not provided" not in msg and "Found entries of length" not in msg:
            sys.__stderr__.write(msg)
    def flush(self): pass

sys.stderr = FilterStderr()

# === Paths
EVAL_DIR = "data/eval"

# === Modelos ASR
ASR_MODELS = {
    "BASE": {"path": "models/base", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT01": {"path": "models/asr_FT01", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT02": {"path": "models/asr_FT02", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT03": {"path": "models/asr_FT03", "vocab": "models/tokenizer_v1_nospecial/vocab.json", "json_vocab": True},
    "FT05": {"path": "models/asr_FT05", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
    "FT08": {"path": "models/asr_FT08", "vocab": "models/tokenizer_v2_base41/vocab.json", "json_vocab": True},
}

# === Modelos de lenguaje
LMS = {
    "LM_BASE": "models/lm_base/modelo_limpio.binary",
    "LM_V01": "models/lm_v01/modelo_finetune.binary",
    "LM_V02": "models/lm_v02/modelo_finetuneV02.binary",
    "LM_V05": "models/lm_v05/modelo_finetune05.binary",
    "LM_V06": "models/lm_v06/modelo_finetune06.binary",
    "LM_V08": "models/lm_v08/modelo_lm_v08.binary",
    "LM_V09": "models/lm_v09/modelo_lm_v09.binary",
}

# === Funciones auxiliares
def load_vocab(path, json_format):
    with open(path, "r", encoding="utf-8") as f:
        if json_format:
            vocab = json.load(f)
            vocab_list = [""] * len(vocab)
            for token, idx in vocab.items():
                vocab_list[idx] = " " if token == "|" else token
        else:
            vocab_list = [line.strip() for line in f]
    return vocab_list

def transcribe(path, model, processor, decoder):
    speech, sr = torchaudio.load(path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
    input_values = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().numpy()
    return decoder.decode(logits).lower()

# === Cargar modelos ASR y decoders LM
print("🔄 Cargando modelos ASR y decoders...")
asr_models = {}
for asr_name, config in ASR_MODELS.items():
    print(f"📦 Cargando ASR: {asr_name}")
    processor = Wav2Vec2Processor.from_pretrained(config["path"])
    model = Wav2Vec2ForCTC.from_pretrained(config["path"])
    vocab_list = load_vocab(config["vocab"], config["json_vocab"])
    asr_models[asr_name] = {"model": model, "processor": processor, "vocab": vocab_list}

decoders = {}
for lm_name, lm_path in LMS.items():
    for asr_name, asr in asr_models.items():
        key = f"{asr_name}__{lm_name}"
        decoders[key] = build_ctcdecoder(asr["vocab"], kenlm_model_path=lm_path)

# === Mostrar transcripciones
print("\n📝 Transcripciones generadas por combinación ASR + LM:\n")
for file in sorted(os.listdir(EVAL_DIR)):
    if not file.endswith(".wav") or "legacy" in file:
        continue

    wav_path = os.path.join(EVAL_DIR, file)
    print(f"\n🎧 Procesando archivo: {file}\n")

    for asr_name, asr in asr_models.items():
        for lm_name in LMS:
            key = f"{asr_name}__{lm_name}"
            print(f"🔄 [ASR: {asr_name}] + [LM: {lm_name}] ...", end=" ")
            pred = transcribe(wav_path, asr["model"], asr["processor"], decoders[key])
            print("✅")
            print(f"→ {pred}\n")

    print("=" * 120)
