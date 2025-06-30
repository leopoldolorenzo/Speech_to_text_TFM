#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune wav2vec2-large-xlsr-53-spanish (dataset_hf)
 · Reanuda desde checkpoint
 · Reduce VRAM en evaluación (eval_accumulation_steps)
"""

import os, logging, torch, evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# ───── Rutas ────────────────────────────────────────────────────────────
BASE_MODEL      = "models/base"
TOKENIZER_DIR   = "models/tokenizer_v2_base41"
DATASET_HF      = "data/common_voice_es/es/fine_tune_08/dataset_hf"
OUTPUT_DIR      = "training/fine_tune_08"
FINAL_MODEL_DIR = "models/fine_tune_08"
LOGGING_DIR     = "logs/fine_tune_08"
os.makedirs(LOGGING_DIR, exist_ok=True)

# ───── Logger ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(f"{LOGGING_DIR}/entrenamiento.log"),
              logging.StreamHandler()],
    level=logging.INFO)
log = logging.getLogger()

# ───── GPU info ────────────────────────────────────────────────────────
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
log.info(f"GPU: {device_name}")

# ───── Carga tokenizer & dataset ───────────────────────────────────────
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
ds_base   = load_from_disk(DATASET_HF)
log.info(f"Dataset base  |  train={len(ds_base['train'])}  val={len(ds_base['validation'])}")

# rename column 'path'→'audio' (si no existe ya)
if "audio" not in ds_base["train"].column_names:
    ds_base = ds_base.rename_column("path", "audio")
    log.info("🔧 Columna 'path' (Audio) renombrada a 'audio'.")

# ───── Pre-procesado (caché en disco) ──────────────────────────────────
PROC_PATH = "data/common_voice_es/es/fine_tune_08/dataset_proc"
if os.path.isdir(PROC_PATH):
    ds = load_from_disk(PROC_PATH)
    log.info(f"Cargado dataset_proc | train={len(ds['train'])} val={len(ds['validation'])}")
else:
    log.info("🔄 Procesando audio → input_values / labels …")
    def prepare(example):
        speech = example["audio"]["array"]
        inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
        with processor.as_target_processor():
            labels = processor(example["sentence"]).input_ids
        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels
        }

    ds = {}
    ds["train"] = ds_base["train"].map(
        prepare, remove_columns=ds_base["train"].column_names,
        num_proc=4, desc="Prep-train")
    ds["validation"] = ds_base["validation"].map(
        prepare, remove_columns=ds_base["validation"].column_names,
        num_proc=2, desc="Prep-val")
    from datasets import DatasetDict
    ds = DatasetDict(ds)
    ds.save_to_disk(PROC_PATH)
    log.info(f"💾 Dataset procesado guardado en {PROC_PATH}")

# ───── Modelo base ─────────────────────────────────────────────────────
cfg   = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=cfg)
model.gradient_checkpointing_enable()

# ───── Métricas ────────────────────────────────────────────────────────
wer = evaluate.load("wer"); cer = evaluate.load("cer")
def metrics_fn(p):
    pred_ids = p.predictions.argmax(-1)
    preds   = processor.batch_decode(pred_ids)
    labels  = processor.batch_decode(p.label_ids, group_tokens=True)
    return {"wer": wer.compute(predictions=preds, references=labels),
            "cer": cer.compute(predictions=preds, references=labels)}

# ───── Data collator ───────────────────────────────────────────────────
from preparar_batches_ctc import PreparadorBatchesCTC
data_collator = PreparadorBatchesCTC(processor=processor, padding=True)

# ───── Último checkpoint ───────────────────────────────────────────────
def last_ckpt(path):
    if not os.path.isdir(path): return None
    ckpts = [c for c in os.listdir(path) if c.startswith("checkpoint-")]
    return os.path.join(path, max(ckpts, key=lambda x:int(x.split("-")[1]))) if ckpts else None
resume_ckpt = last_ckpt(OUTPUT_DIR)
log.info("🔄 Reanudando desde checkpoint: %s", resume_ckpt or "None (inicio)")

# ───── TrainingArguments (cambios marcados) ────────────────────────────
args = TrainingArguments(
    output_dir           = OUTPUT_DIR,
    per_device_train_batch_size = 6,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 1,
    evaluation_strategy  ="steps",       # ←  sigue igual
    eval_steps           = 2000,
    save_steps           = 2000,
    eval_accumulation_steps = 10,        # ← **libera VRAM cada 10 miniciclos**
    # evaluación cada época si lo prefieres:
    # evaluation_strategy="epoch",
    num_train_epochs     = 20,
    fp16                 = torch.cuda.is_available(),
    learning_rate        = 2e-5,
    lr_scheduler_type    ="cosine",
    warmup_ratio         = 0.1,
    save_total_limit     = 2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better    = False,
    logging_dir          = LOGGING_DIR,
    logging_steps        = 50,
    report_to            ="tensorboard"
)

# ───── Trainer ─────────────────────────────────────────────────────────
trainer = Trainer(
    model             = model,
    args              = args,
    tokenizer         = processor,
    data_collator     = data_collator,
    train_dataset     = ds["train"],
    eval_dataset      = ds["validation"],
    compute_metrics   = metrics_fn,
    callbacks         =[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train(resume_from_checkpoint=resume_ckpt)

# ───── Guardar modelo final ────────────────────────────────────────────
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"✅ Modelo guardado en {FINAL_MODEL_DIR}")
