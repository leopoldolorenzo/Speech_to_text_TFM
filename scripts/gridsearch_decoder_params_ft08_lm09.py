import os
import csv
from pathlib import Path
import torch
import torchaudio
import evaluate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

# === Configuración ===
ASR_PATH = "models/asr_FT08"
VOCAB_PATH = "models/asr_FT08/vocab.json"
LM_PATH = "models/lm_v09/modelo_lm_v09.binary"
AUDIO_DIR = Path("data/eval_confiable_100/wav")
TXT_DIR = Path("data/eval_confiable_100/txt")
RESULTS_CSV = Path("results/gridsearch_ft08_lm09.csv")

# Parámetros a evaluar
ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]
BETA_VALUES = [0.0, 0.5, 1.0]
BEAM_WIDTHS = [25, 50, 100]

# Cargar métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# Función para cargar vocabulario
def load_vocab(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_list = [""] * len(vocab)
    for token, idx in vocab.items():
        vocab_list[idx] = " " if token == "|" else token
    return vocab_list

# Cargar modelo y tokenizer
print("▶️  Cargando ASR …")
processor = Wav2Vec2Processor.from_pretrained(ASR_PATH, local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained(ASR_PATH, local_files_only=True)
model.eval()

# Cargar vocabulario
vocab_list = load_vocab(VOCAB_PATH)

# Cargar referencias
audios = sorted(AUDIO_DIR.glob("*.wav"))
referencias = {}
for audio_path in audios:
    txt_path = TXT_DIR / audio_path.with_suffix(".txt").name
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            referencias[audio_path.name] = f.read().strip().lower()

# === Grid Search ===
results = []
for alpha in ALPHA_VALUES:
    for beta in BETA_VALUES:
        for beam_width in BEAM_WIDTHS:
            print(f"\n🔎 Evaluando: alpha={alpha}, beta={beta}, beam_width={beam_width}")
            decoder = build_ctcdecoder(
                vocab_list,
                kenlm_model_path=LM_PATH,
                alpha=alpha,
                  beta=beta,
            )

            all_preds = []
            all_refs = []

            for audio_path in audios:
                ref = referencias.get(audio_path.name)
                if not ref:
                    continue

                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits[0].cpu().numpy()

                pred = decoder.decode(logits, beam_width=beam_width).lower()
                all_preds.append(pred)
                all_refs.append(ref)

            score_wer = wer.compute(predictions=all_preds, references=all_refs)
            score_cer = cer.compute(predictions=all_preds, references=all_refs)
            print(f"✔️ WER: {score_wer:.3f} | CER: {score_cer:.3f}")

            results.append([alpha, beta, beam_width, round(score_wer, 4), round(score_cer, 4)])

# Guardar resultados
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["alpha", "beta", "beam_width", "WER", "CER"])
    writer.writerows(results)

print(f"\n✅ Grid search completado. Resultados guardados en {RESULTS_CSV}")
