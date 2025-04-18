# ✅ scripts/train_model_fast_continue.py (continúa desde checkpoint, optimizado para velocidad)

import os
import torch
import numpy as np
import evaluate
import shutil
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from data_collator_ctc import DataCollatorCTCWithPadding

# === CONFIGURACIÓN ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
DATASET_PATH = "data/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1"
OUTPUT_DIR = "training/spanish_w2v2_v1_gpu12gb"
FINAL_MODEL_DIR = "models/spanish_w2v2_v1"

print("[✅] Entorno disponible.")
print("[📂] Cargando dataset reducido...")
dataset = load_from_disk(DATASET_PATH)

print("[🔤] Cargando processor desde:", TOKENIZER_DIR)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

# === Limpiar memoria GPU
torch.cuda.empty_cache()

# === Preprocesamiento ===
def prepare(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("[🔄] Preprocesando datos...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names, num_proc=4)

# === Data Collator ===
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# === Métrica: WER
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# === Cargar modelo base
print("[🧠] Cargando modelo base...")
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    vocab_size=len(processor.tokenizer),
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    ignore_mismatched_sizes=True,
)

# === Optimización memoria y velocidad
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# === Argumentos de entrenamiento optimizados
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=False,
    per_device_train_batch_size=4,        # 🔽 bajamos para aliviar la VRAM
    per_device_eval_batch_size=4,         # 🔽 mismo en validación
    gradient_accumulation_steps=8,        # 🔼 para mantener batch efectivo ~32
    evaluation_strategy="steps",
    eval_steps=250,                       # ✅ evaluación frecuente
    save_strategy="steps",
    save_steps=250,                       # ✅ checkpoint frecuente
    save_total_limit=5,                   # ✅ mantiene últimos 5
    learning_rate=3e-4,
    warmup_steps=500,
    num_train_epochs=10,
    fp16=True,                            # ✅ aceleración por hardware
    logging_dir="logs",
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# === Inicializar Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.tokenizer,
)

trainer.args.group_by_length = False  # ⚠️ Asegura compatibilidad al reanudar

# === Detectar y continuar desde último checkpoint
latest_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = os.path.join(
            OUTPUT_DIR,
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"[⏸️] Reanudando desde checkpoint: {latest_checkpoint}")
    else:
        print("[🚀] Iniciando desde cero.")
else:
    print("[🚀] Iniciando desde cero.")

# === Entrenamiento
print("[🔥] Continuando entrenamiento...")
trainer.train(resume_from_checkpoint=latest_checkpoint)

# === Guardado final
print("[💾] Guardando modelo final en:", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# === Copia a carpeta de modelos
if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
shutil.copytree(OUTPUT_DIR, FINAL_MODEL_DIR)
print(f"[📦] Modelo final copiado a: {FINAL_MODEL_DIR}")
print("[✅] Entrenamiento completado.")
