import os
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diagnose")

# === CONFIGURACIÓN ===
MODEL_DIR = "models/debug_w2v2"
VAL_TSV = "data/val_dataset.tsv"
AUDIO_DIR = "data/eval"
DATASET_DIR = "data/common_voice_subset"

# === 1. Verificar vocabulario y tokenizador ===
try:
    vocab_path = Path(MODEL_DIR) / "vocab.json"
    tokenizer_path = Path(MODEL_DIR) / "tokenizer_config.json"
    vocab = json.load(open(vocab_path))
    tokenizer_config = json.load(open(tokenizer_path))

    special_tokens = [t for t in vocab if t in ["[PAD]", "[UNK]", "<s>", "</s>"]]
    delimiter = tokenizer_config.get("word_delimiter_token", None)

    log.info(f"🔤 Tamaño del vocabulario: {len(vocab)}")
    log.info(f"🧩 Tokens especiales detectados: {special_tokens}")
    log.info(f"🔗 Delimitador de palabras: {delimiter}")
except Exception as e:
    log.error(f"❌ Error cargando vocab/tokenizer: {e}")

# === 2. Verificar etiquetas ===
try:
    val_data = load_dataset("csv", data_files=VAL_TSV, delimiter="\t")['train']
    valid_chars = set("".join(vocab).replace("[PAD]", "").replace("[UNK]", "").replace("<s>", "").replace("</s>", "").replace("|", " "))

    invalid_samples = []
    for ex in val_data:
        text = ex['sentence'].lower().strip()
        if not text:
            continue
        if any(c not in valid_chars for c in text if c != " "):
            invalid_samples.append(text)

    log.info(f"🧪 Transcripciones fuera de vocabulario: {len(invalid_samples)} (ej. '{invalid_samples[:1]}')")
except Exception as e:
    log.error(f"❌ Error verificando etiquetas: {e}")

# === 3. Verificar el modelo ===
try:
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
    lm_head_size = model.lm_head.out_features
    expected_vocab_size = len(vocab)
    if lm_head_size != expected_vocab_size:
        log.warning(f"⚠️ Tamaño de lm_head ({lm_head_size}) no coincide con vocab ({expected_vocab_size})")
    else:
        log.info(f"✅ lm_head coincide con vocab: {lm_head_size}")
except Exception as e:
    log.error(f"❌ Error cargando modelo: {e}")

# === 4. Verificar archivos de audio ===
try:
    bad_audio = []
    for wav in Path(AUDIO_DIR).glob("*.wav"):
        info = sf.info(str(wav))
        if info.samplerate != 16000 or info.channels != 1:
            bad_audio.append((wav.name, info.samplerate, info.channels))

    if bad_audio:
        for f, sr, ch in bad_audio:
            log.warning(f"🎧 Archivo '{f}' - SR: {sr}, Channels: {ch}")
    else:
        log.info("🎧 Todos los audios tienen formato correcto (mono, 16kHz)")
except Exception as e:
    log.error(f"❌ Error analizando audios: {e}")

# === 5. Tamaño del dataset ===
try:
    train_arrows = list(Path(DATASET_DIR + "/train").glob("*.arrow"))
    test_arrows = list(Path(DATASET_DIR + "/test").glob("*.arrow"))
    log.info(f"📊 Train shards: {len(train_arrows)} | Test shards: {len(test_arrows)}")
except Exception as e:
    log.error(f"❌ Error contando shards: {e}")
