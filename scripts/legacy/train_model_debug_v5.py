# train_model_debug_v5.py
# -------------------------------------------------------
# Entrenamiento reducido para debugging (1000 ejemplos)
# -------------------------------------------------------

import os, re, shutil, torch, logging
import numpy as np
import evaluate
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
)
from data_collator_ctc import DataCollatorCTCWithPadding

# === CONFIGURACIÓN ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
ORIGINAL_DATASET_PATH = "data/common_voice_subset_fixed/common_voice_subset"
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"
OUTPUT_DIR = "training/debug_w2v2"
FINAL_MODEL_DIR = "models/debug_w2v2"
LOGGING_DIR = "logs_debug"

# === LOGGING ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-záéíóúñü ]+", "", text)

def prepare(batch):
    try:
        text = clean_text(batch.get("sentence", ""))
        if not text.strip():
            log.warning(f"❌ Texto vacío tras limpieza: '{batch.get('sentence', '')}'")
            return None
        audio = processor(batch["audio"]["array"], sampling_rate=16000)
        with processor.as_target_processor():
            labels = processor.tokenizer(text).input_ids
        if not labels:
            log.warning(f"⚠️ Etiquetas vacías para: '{text}'")
            return None
        return {
            "input_values": audio["input_values"][0],
            "attention_mask": audio["attention_mask"][0],
            "labels": labels
        }
    except Exception as e:
        log.error(f"💥 Error procesando ejemplo: {e}")
        return None

def compute_metrics(pred):
    try:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        result = {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}
        return result
    except Exception as e:
        log.error(f"Error en métricas: {e}")
        return {"wer": float("inf")}

class ConsoleLogger(TrainerCallback):
    def __init__(self, interval): self.interval = interval
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and state.global_step % self.interval == 0 and state.global_step > 0:
            log.info(" | ".join([f"{k.upper()}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]))

# === 1. CARGA DE DATASET REDUCIDO PARA DEBUG ===
assert os.path.exists(ORIGINAL_DATASET_PATH), "Dataset no encontrado"
dataset = load_from_disk(ORIGINAL_DATASET_PATH)
dataset = DatasetDict({
    "train": dataset["train"].select(range(1000)),
    "test": dataset["test"].select(range(200))
})

processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
torch.cuda.empty_cache()

log.info("🔄 Preprocesando dataset...")
map_args = {"remove_columns": dataset["train"].column_names}
dataset = dataset.map(prepare, **map_args).filter(lambda x: x is not None and len(x["labels"]) > 0)

log.info(f"🧪 Ejemplos listos: Train = {len(dataset['train'])} | Test = {len(dataset['test'])}")

# === 2. CONFIGURACIÓN DEL MODELO ===
log.info("🧠 Cargando modelo...")
vocab_size = len(processor.tokenizer)
config = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=vocab_size)
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config, ignore_mismatched_sizes=True)
model.gradient_checkpointing_enable()
# ⚠️ No congelamos el encoder
# model.freeze_feature_encoder()  ← ¡No activar!

# === 3. VALIDACIÓN DE BATCH ===
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)
batch = next(iter(loader))
log.info(f"Batch OK | input: {batch['input_values'].shape} | labels: {batch['labels'].shape}")
for i in range(min(3, len(batch["labels"]))):
    log.info(f"🔎 Ejemplo {i+1} de labels: {batch['labels'][i].tolist()}")

# === 4. MÉTRICAS Y ENTRENAMIENTO ===
wer_metric = evaluate.load("wer")

# === Cálculo dinámico de warmup ===
train_batch_size = 2
gradient_accumulation = 2
num_train_epochs = 3
num_train_examples = len(dataset["train"])

steps_per_epoch = num_train_examples // train_batch_size // gradient_accumulation
total_steps = steps_per_epoch * num_train_epochs
warmup_steps = max(10, int(0.1 * total_steps))  # al menos 10 pasos

log.info(f"🚦 Steps por epoch: {steps_per_epoch} | Total steps: {total_steps} | Warmup steps: {warmup_steps}")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    learning_rate=1e-4,
    num_train_epochs=num_train_epochs,
    warmup_steps=warmup_steps,  # ✅ valor dinámico calculado arriba
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    logging_first_step=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=processor,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[ConsoleLogger(10)]
)

log.info("🚀 Entrenamiento DEBUG iniciado...")
trainer.train()

log.info("💾 Guardando modelo debug...")
trainer.save_model(FINAL_MODEL_DIR)

# ❗ Eliminar tokens BOS/EOS que rompen la decodificación
processor.tokenizer.bos_token = None
processor.tokenizer.eos_token = None

processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"✅ Guardado en: {FINAL_MODEL_DIR}")
