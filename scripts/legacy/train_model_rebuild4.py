# train_model_rebuild4.py
# ---------------------------------------------------------
# Entrenamiento de Wav2Vec2 con tokenizer personalizado.
# Utiliza processor limpio construido manualmente.
# ---------------------------------------------------------

import os, re, shutil, torch, logging
import numpy as np
import evaluate
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
)
from data_collator_ctc import DataCollatorCTCWithPadding

# === CONFIGURACIÓN ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
ORIGINAL_DATASET_PATH = "data/common_voice_subset"
PROCESSED_DATASET_PATH = "data/common_voice_subset_processed"
TOKENIZER_DIR = "models/tokenizer_v1_nospecial"
OUTPUT_DIR = "training/spanish_w2v2_rebuild"
FINAL_MODEL_DIR = "models/spanish_w2v2_rebuild"
LOGGING_DIR = "logs_rebuild"

# === OPCIONES ===
ENABLE_PARALLEL_MAP = False
ENABLE_ENCODER_FREEZE = True
ENABLE_CER_METRIC = True
KEEP_PUNCTUATION = False
LOG_INTERVAL = 25

# === LOGGING ===
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger()

# === FUNCIONES AUXILIARES ===
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-záéíóúñü .!?]" if KEEP_PUNCTUATION else r"[^a-záéíóúñü ]+", "", text)

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
        if cer_metric:
            result["cer"] = cer_metric.compute(predictions=pred_str, references=label_str)
        return result
    except Exception as e:
        log.error(f"Error en métricas: {e}")
        return {"wer": float("inf"), "cer": float("inf") if cer_metric else None}

class ConsoleLogger(TrainerCallback):
    def __init__(self, interval): self.interval = interval
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and state.global_step % self.interval == 0 and state.global_step > 0:
            log.info(" | ".join([f"{k.upper()}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]))

# === 1. CARGA DE DATASET ===
assert os.path.exists(ORIGINAL_DATASET_PATH), "Dataset no encontrado"
dataset = load_from_disk(ORIGINAL_DATASET_PATH)
assert "train" in dataset and "test" in dataset
assert all("audio" in x and "sentence" in x for x in dataset["train"].select(range(3)))

processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
torch.cuda.empty_cache()

log.info("🔄 Preprocesando dataset...")
map_args = {"remove_columns": dataset["train"].column_names}
if ENABLE_PARALLEL_MAP: map_args["num_proc"] = 4
dataset = dataset.map(prepare, **map_args).filter(lambda x: x is not None and len(x["labels"]) > 0)

# 🔍 NUEVO LOG: Verificación post-filtrado
log.info(f"🧮 Ejemplos válidos tras preprocesamiento:")
log.info(f"   ▶️ Train: {len(dataset['train'])}")
log.info(f"   🧪 Test : {len(dataset['test'])}")
assert len(dataset["train"]) > 0, "⛔ Dataset de entrenamiento quedó vacío"
assert len(dataset["test"]) > 0, "⛔ Dataset de test quedó vacío"

if os.path.exists(PROCESSED_DATASET_PATH):
    log.warning(f"🗑️ Eliminando dataset previo: {PROCESSED_DATASET_PATH}")
    shutil.rmtree(PROCESSED_DATASET_PATH)
dataset.save_to_disk(PROCESSED_DATASET_PATH)

# === 2. CONFIGURACIÓN DEL MODELO ===
log.info("🧠 Cargando modelo...")
vocab_size = len(processor.tokenizer)
assert 0 < vocab_size < 2000
config = Wav2Vec2Config.from_pretrained(BASE_MODEL, vocab_size=vocab_size)
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config, ignore_mismatched_sizes=True)
if ENABLE_ENCODER_FREEZE: model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# === 3. VALIDACIÓN DE BATCH ===
log.info("🧪 Validando batch...")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)
batch = next(iter(loader))
log.info(f"Batch OK | input: {batch['input_values'].shape} | labels: {batch['labels'].shape}")
for i in range(min(3, len(batch["labels"]))):
    log.info(f"🔎 Ejemplo {i+1} de labels: {batch['labels'][i].tolist()}")

# === 4. MÉTRICAS Y TRAINING ===
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer") if ENABLE_CER_METRIC else None

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    eval_steps=250,
    save_steps=250,
    save_total_limit=3,
    learning_rate=3e-4,
    num_train_epochs=5,
    warmup_steps=0,
    fp16=torch.cuda.is_available(),
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
    callbacks=[ConsoleLogger(LOG_INTERVAL)]
)

# === 5. REANUDAR DESDE CHECKPOINT (SI EXISTE) ===
ckpt = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = sorted([c for c in os.listdir(OUTPUT_DIR) if c.startswith("checkpoint-")], key=lambda x: int(x.split("-")[-1]))
    if ckpts:
        ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])
        log.info(f"⏯️ Reanudando desde checkpoint: {ckpt}")
    else:
        log.info("🚀 Iniciando desde cero.")
else:
    log.info("🚀 Carpeta nueva. Iniciando desde cero.")

# === 6. ENTRENAMIENTO ===
log.info("🔥 Comenzando entrenamiento...")
log.info(f"🔍 Total training steps esperados: {trainer.state.max_steps}")
log.info(f"🚦 Warmup steps: {training_args.warmup_steps}")
trainer.train(resume_from_checkpoint=ckpt)

# === 7. GUARDADO DEL MODELO ===
log.info("💾 Guardando modelo...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
log.info(f"📦 Guardado en: {FINAL_MODEL_DIR}")
