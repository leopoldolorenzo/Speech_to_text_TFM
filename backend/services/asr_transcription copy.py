# backend/services/asr_transcription.py

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from pydub import AudioSegment
import json

# Ajusta las rutas a tus modelos
ASR_PATH = "models/asr_FT08"
VOCAB_PATH = "models/asr_FT08/vocab.json"
LM_PATH = "models/lm_v09/modelo_lm_v09.binary"

BEAM_WIDTH = 50

# Cargar vocabulario
def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = [""] * len(vocab_json)
    for token, idx in vocab_json.items():
        vocab[idx] = " " if token == "|" else token
    return vocab

# Inicializar modelos al inicio para ahorrar memoria
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = load_vocab(VOCAB_PATH)
processor = Wav2Vec2Processor.from_pretrained(ASR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(ASR_PATH).to(DEVICE).eval()
decoder = build_ctcdecoder(vocab, kenlm_model_path=LM_PATH, alpha=0.5, beta=0.0)

def transcribe_segment(segment):
    """
    Transcribe un segmento de audio (PyDub) con Wav2Vec2 + LM.
    """
    # Recorta el segmento
    clip = AudioSegment.from_wav(segment["path"])
    clip_segment = clip[int(segment["start"]*1000): int(segment["end"]*1000)]

    # Convertir a np.float32 normalizado
    samples = np.array(clip_segment.get_array_of_samples()).astype(np.float32) / 32768.0

    # Tokenizar y transcribir
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(inputs).logits[0].cpu().numpy()

    # Decodificar con KenLM
    transcription = decoder.decode(logits, beam_width=BEAM_WIDTH)

    return transcription.strip()
