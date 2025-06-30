# scripts/entrenar_finetune_05.py

import os, re, shutil, torch, logging
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl,
    EarlyStoppingCallback
)
from preparar_batches_ctc import PreparadorBatchesCTC

# === Configuración de rutas ===
BASE_MODEL = "models/base"
TOKENIZER_DIR = "models/tokenizer_v2_base41"
DATASET_PATH = "data/common_voice_es/es/fine_tune_05/dataset_hf"
OUTPUT_DIR = "training/fine_tune_05"
FINAL_MODEL_DIR = "models/fine_tune_05"
LOGGING_DIR = "logs/fine_tune_05"

# === Logger ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9, 0)

# === Último checkpoint
def get_last_checkpoint_manual(output_dir):
    if not os.path.isdir(output_dir): return None
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    return max(checkpoints, key=lambda x: int(x.split("-")[-1])) if checkpoints else None

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-záéíóúñü ]+", "", text)

def prepare(batch):
    try:
        text = clean_text(batch["text"])
        if not text:
            return None
        audio = processor(batch["audio"]["array"], sampling_rate=16000)
        with processor.as_target_processor():
            labels = processor.tokenizer(text).input_ids
        if not labels:
            return None
        return {
            "input_values": audio["input_values"][0],
            "attention_mask": audio["attention_mask"][0],
            "labels": labels
        }
    except Exception as e:
        log.error(f"Error procesando muestra: {e}")
        return None

# === Métricas
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=True)
    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str)
    }

class ConsoleLogger(TrainerCallback):
    def __init__(self, interval): self.interval = interval
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and state.global_step % self.interval == 0 and state.global_step > 0:
            log.info(" | ".join([f"{k.upper()}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]))

# === Dataset
log.info("🔄 Cargando dataset...")
dataset = load_from_disk(DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

log.info("🔄 Preprocesando dataset...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names).filter(lambda x: x is not None)
log.info(f"🧪 Dataset listo: Train = {len(dataset['train'])} | Validation = {len(dataset['validation'])}")

# === Checkpoint o reinicio
checkpoint_path = get_last_checkpoint_manual(OUTPUT_DIR)
if checkpoint_path:
    log.info(f"📍 Reanudando desde checkpoint: {checkpoint_path}")
else:
    if os.path.exists(OUTPUT_DIR):
        log.warning(f"🧹 Eliminando entrenamiento anterior: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

# === Modelo base
log.info("🧠 Cargando modelo base...")
config = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config)
model.gradient_checkpointing_enable()

# === Hiperparámetros
train_batch_size = 4
eval_batch_size = 1
gradient_accumulation = 2
epochs = 20

train_size = len(dataset["train"])
steps_per_epoch = max(1, train_size // (train_batch_size * gradient_accumulation))
total_steps = steps_per_epoch * epochs
warmup_steps = max(10, int(total_steps * 0.1))

log.info(f"📊 max_steps: {total_steps} | warmup_steps: {warmup_steps}")

data_collator = PreparadorBatchesCTC(processor=processor, padding=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    eval_accumulation_steps=5,
    num_train_epochs=epochs,
    max_steps=total_steps,
    learning_rate=3e-5,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_dir=LOGGING_DIR,
    logging_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=processor,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[
        ConsoleLogger(50),
        EarlyStoppingCallback(early_stopping_patience=10)
    ]
)

# === Entrenamiento
log.info("🚀 Iniciando entrenamiento fine_tune_05...")
trainer.train(resume_from_checkpoint=checkpoint_path if checkpoint_path else None)

# === Guardar modelo final
log.info("💾 Guardando modelo final...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"✅ Modelo guardado en: {FINAL_MODEL_DIR}")
