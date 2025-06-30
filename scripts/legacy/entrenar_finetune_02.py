# entrenar_finetune_02.py
# Continúa el fine-tuning del modelo wav2vec2 usando nuevos datos (fine_tune_02), partiendo desde el modelo fine_tune_01.

import os, re, shutil, torch, logging
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
)
from preparar_batches_ctc import PreparadorBatchesCTC

# === Configuración de paths ===
BASE_MODEL = "models/fine_tune_01"  # ⬅️ Aquí usamos el modelo ya fine-tuneado
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"
DATASET_PATH = "data/common_voice_es/es/fine_tune_02/dataset_hf"
OUTPUT_DIR = "training/fine_tune_02"
FINAL_MODEL_DIR = "models/fine_tune_02"
LOGGING_DIR = "logs/fine_tune_02"

# === Inicializar logging ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9, 0)

# === Normalizador de texto ===
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-záéíóúñü ]+", "", text)

def prepare(batch):
    try:
        text = clean_text(batch["sentence"])
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

# === Métricas ===
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

# === Cargar dataset y processor ===
log.info("🔄 Cargando dataset desde disco...")
dataset = load_from_disk(DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

log.info("🔄 Preprocesando dataset...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names).filter(lambda x: x is not None)
log.info(f"🧪 Dataset listo: Train = {len(dataset['train'])} | Test = {len(dataset['test'])}")

if os.path.exists(OUTPUT_DIR):
    log.warning(f"🧹 Eliminando checkpoints previos de: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

# === Cargar modelo anterior (fine_tune_01) ===
log.info("🧠 Cargando modelo fine-tuneado anterior...")
config = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config)
model.gradient_checkpointing_enable()

# === Hiperparámetros y entrenamiento ===
train_batch_size = 4
eval_batch_size = 2
gradient_accumulation = 8
epochs = 5
train_size = len(dataset["train"])
steps_per_epoch = train_size // (train_batch_size * gradient_accumulation)
total_steps = steps_per_epoch * epochs
warmup_steps = max(50, int(total_steps * 0.1))

log.info(f"📊 total_steps: {total_steps} | warmup_steps: {warmup_steps}")

data_collator = PreparadorBatchesCTC(processor=processor, padding=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=train_batch_size,         # ⚙️ Batch size ajustado para tu GPU
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation,    # ⏫ Para simular batch grande
    evaluation_strategy="steps",
    eval_steps=1000,                                      # ✅ Evaluar cada 1000 steps (más espaciado)
    save_steps=1000,                                      # 💾 Guardar checkpoints cada 1000 steps
    num_train_epochs=5,                                   # 🧠 Igual que el fine_tune_01
    learning_rate=1e-4,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",                           # 🚀 Evita que el LR se quede en 0
    fp16=True,                                             # ✅ Activado si tenés soporte
    logging_dir=LOGGING_DIR,
    logging_steps=50,                                     # 🪵 Log cada 50 pasos
    save_total_limit=2,                                   # 💾 Máx. 2 checkpoints guardados
    load_best_model_at_end=True,
    metric_for_best_model="wer",                          # 📉 Guardar el de menor WER
    greater_is_better=False,
    report_to="tensorboard"                               # 📊 Compatible con Visual Studio Code
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=processor,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[ConsoleLogger(25)]
)

log.info("🚀 Iniciando entrenamiento incremental (fine_tune_02)...")
trainer.train()

log.info("💾 Guardando modelo final...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"✅ Modelo final guardado en: {FINAL_MODEL_DIR}")
