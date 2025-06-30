#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune wav2vec2-large-xlsr-53-spanish utilizando dataset_hf ya procesado.
Evita reprocesar el TSV y aprovecha el dataset guardado con audio + texto listos.
"""

import os
import logging
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# === Rutas ===
BASE_MODEL = "models/base"
TOKENIZER_DIR = "models/tokenizer_v2_base41"
DATASET_HF_PATH = "data/common_voice_es/es/fine_tune_08/dataset_hf"
OUTPUT_DIR = "training/fine_tune_08"
FINAL_MODEL_DIR = "models/fine_tune_08"
LOGGING_DIR = "logs/fine_tune_08"

# === Logger ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

# === Cargar processor y dataset
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
ds = load_from_disk(DATASET_HF_PATH)
log.info(f"📊 Dataset cargado desde {DATASET_HF_PATH} | train={len(ds['train'])} | validation={len(ds['validation'])}")

# === Configurar modelo
cfg = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=cfg)
model.gradient_checkpointing_enable()

# === Métricas
wer = evaluate.load("wer")
cer = evaluate.load("cer")

def metrics(p):
    pred_ids = p.predictions.argmax(-1)
    pred = processor.batch_decode(pred_ids)
    labels = processor.batch_decode(p.label_ids, group_tokens=True)
    return {
        "wer": wer.compute(predictions=pred, references=labels),
        "cer": cer.compute(predictions=pred, references=labels)
    }

# === Data collator
from preparar_batches_ctc import PreparadorBatchesCTC
data_collator = PreparadorBatchesCTC(processor=processor, padding=True)

# === Hiperparámetros
train_bs, grad_accum = 6, 1
steps_per_epoch = len(ds["train"]) // (train_bs * grad_accum)
epochs = 20
total_steps = steps_per_epoch * epochs
warmup = max(10, int(total_steps * 0.1))

# === TrainingArguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=train_bs,
    gradient_accumulation_steps=grad_accum,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    num_train_epochs=epochs,
    max_steps=total_steps,
    warmup_steps=warmup,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    report_to="tensorboard",
)

# === Trainer
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=processor,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# === Entrenamiento
log.info("🚀 Iniciando entrenamiento...")
trainer.train()

# === Guardar modelo final
log.info("💾 Guardando modelo final...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"✅ Modelo guardado en: {FINAL_MODEL_DIR}")
