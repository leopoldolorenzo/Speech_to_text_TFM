import os
import re
import torch
import logging
import numpy as np
import evaluate
import shutil
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from data_collator_ctc import DataCollatorCTCWithPadding

# === CONFIGURACIÓN DE LOGS ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

# === CONFIGURACIÓN ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
DATASET_PATH = "data/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1"
OUTPUT_DIR = "training/debug_w2v2"
LOGGING_DIR = "logs_debug"

# === FUNCIONES ===
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúñü ]+", "", re.sub(r"\s+", " ", text))
    return text

def prepare(batch):
    try:
        text = clean_text(batch["sentence"])
        if not text:
            return None
        audio = processor(batch["audio"]["array"], sampling_rate=16000)
        labels = processor.tokenizer(text).input_ids
        if not labels:
            return None
        return {
            "input_values": audio["input_values"][0],
            "attention_mask": audio["attention_mask"][0],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    except Exception as e:
        log.error(f"Error preparando ejemplo: {e}")
        return None

def compute_metrics(pred):
    try:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = evaluate.load("wer")
        return {"wer": wer.compute(predictions=pred_str, references=label_str)}
    except Exception as e:
        log.error(f"Error en compute_metrics: {e}")
        return {"wer": float("inf")}

class ConsoleLogger(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            step = state.global_step
            msg = f"[Step {step}] " + " | ".join(f"{k.upper()}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))
            log.info(msg)

# === CARGA DE DATOS Y PROCESSOR ===
log.info("Cargando dataset y processor...")
assert os.path.exists(DATASET_PATH), "Dataset no encontrado"
dataset = load_from_disk(DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

# === REDUCIR EL DATASET PARA DEBUG ===
log.info("Reduciendo dataset para debug (200 ejemplos)...")
dataset = DatasetDict({
    "train": dataset["train"].select(range(100)),
    "test": dataset["test"].select(range(100))
})

log.info("Preprocesando ejemplos...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names)
dataset = dataset.filter(lambda x: x is not None and len(x["labels"]) > 0)

# === CONFIGURACIÓN DEL MODELO ===
log.info("Configurando modelo...")
config = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config, ignore_mismatched_sizes=True)
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# === DATA COLLATOR ===
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# === TRAINING ARGUMENTS ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=20,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    report_to="none",
    log_level="info",
    disable_tqdm=False
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[ConsoleLogger()]
)

# === ENTRENAR ===
log.info("Entrenando modelo (debug)...")
trainer.train()

# === GUARDAR ===
log.info("Guardando modelo (debug)...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
log.info(f"Modelo guardado en {OUTPUT_DIR}")