# probar_con_decoder_lm.py
# Compara la salida greedy vs LM decoder con ctcdecode

import torch
import random
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from ctcdecode import CTCBeamDecoder

# === Configuración ===
DATASET_DIR = "data/common_voice_es/es/fine_tune_01/dataset_hf"
MODEL_DIR = "models/fine_tune_01"
VOCAB_PATH = "data/vocab/vocab.txt"
LM_PATH = "data/lm/modelo_limpio.bin"

# === Cargar tokenizer vocabulario
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_list = [line.strip() for line in f.readlines()]
# Asegurar que el vocab tiene un "blank" token al inicio
labels = [""] + vocab_list

# === Decoder LM
decoder = CTCBeamDecoder(
    labels,
    model_path=LM_PATH,
    alpha=0.5,
    beta=1.0,
    beam_width=100,
    num_processes=4,
    blank_id=0,
    log_probs_input=True
)

# === Cargar modelo y processor
print(f"📦 Cargando modelo y processor desde: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

# === Muestra aleatoria del dataset
dataset = load_from_disk(DATASET_DIR)
sample = random.choice(dataset["test"])
audio_input = sample["audio"]["array"]
ground_truth = sample["sentence"]

# === Procesar entrada
inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

# === Greedy decoding
pred_ids = torch.argmax(log_probs, dim=-1)
greedy_text = processor.batch_decode(pred_ids)[0]

# === LM decoding
beam_result, _, _, out_lens = decoder.decode(log_probs)

# Convertir resultado LM a string
lm_text = "".join([labels[i] for i in beam_result[0][0][:out_lens[0][0]]])

# === Mostrar resultados
print("\n🔍 Ejemplo de prueba con LM")
print(f"REF    : {ground_truth}")
print(f"GREEDY : {greedy_text}")
print(f"LM     : {lm_text}")
