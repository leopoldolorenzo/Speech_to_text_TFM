Perfecto, aquí tienes:

---

## 📁 Archivos a crear

- `~/TFM/.gitignore`
- `~/TFM/README.md`

---

## ✅ `.gitignore` completo

Crea el archivo `~/TFM/.gitignore` con este contenido:

```gitignore
# === ENTORNOS ===
.env
.venv/
__pycache__/
*.pyc

# === MODELOS PESADOS ===
models/*
!models/base/  # permite base si es ligero

# === AUDIOS ===
audios/**/*.wav

# === LOGITS y SALIDAS ===
outputs/**
results/**/*.csv

# === LENGUAJE Y CORPUS ===
data/lm/*.txt
data/lm/*.arpa
data/lm/*.bin

# === ARCHIVOS BINARIOS O GRANDES ===
*.npy
*.bin
*.pt
*.pkl
*.zip
*.gz
```

---

## 📝 `README.md` completo

Guarda esto como `~/TFM/README.md`:

```markdown
# 🧠 Proyecto ASR - Transcripción de voz a texto en Español

Este proyecto implementa un sistema de reconocimiento automático del habla (ASR) utilizando **Wav2Vec2.0**, **KenLM** y decodificación con modelo de lenguaje (LM) para mejorar la precisión de las transcripciones.

---

## 📂 Estructura del proyecto

```bash
TFM/
├── audios/                  # Audios originales y convertidos (.wav)
│   └── convertidos/
├── data/
│   ├── vocab/               # vocab.txt y lexicon.txt
│   ├── lm/                  # Corpus y modelo de lenguaje (KenLM)
│   └── val_dataset.tsv      # Dataset de evaluación
├── models/                  # Modelos base y fine-tuned (NO subidos por defecto)
├── outputs/                 # Logits generados por los modelos
├── results/                 # Métricas y transcripciones
├── scripts/                 # Todos los scripts Python del sistema
├── notebooks/               # (Opcional) análisis y visualización
├── .gitignore
└── README.md
```

---

## 🚀 Características

- 📌 Usa `Wav2Vec2.0` (modelo HuggingFace en español)
- 📚 Mejora con decodificación `beam search` + `KenLM`
- 🔎 Métricas de calidad: **WER** y **CER**
- 🔁 Soporte multimodelo (base + fine-tuned)
- 🖥️ Scripts listos para:
  - Procesar audios (`pipeline_asr.py`)
  - Evaluar modelo (`evaluate_model.py`)
  - Decodificación con LM (`decode_with_lm.py`)
  - Extraer vocabulario y logits

---

## 🛠️ Requisitos

- Python 3.10
- Conda / Mamba
- PyTorch
- HuggingFace Transformers
- Librosa, KenLM, jiwer

---

## ▶️ Cómo empezar

```bash
# 1. Activar entorno
conda activate asr_env

# 2. Procesar un audio
python scripts/pipeline_asr.py audios/convertidos/prueba01.wav

# 3. Evaluar el sistema completo
python scripts/evaluate_model.py
```

---

## 🧪 Evaluación

Calcula métricas de calidad comparando las transcripciones reales con las predichas.

```bash
# Dataset de evaluación
data/val_dataset.tsv

# Resultado:
results/metrics.csv
```

---

## 📦 Créditos

- Modelo base: [`jonatasgrosman/wav2vec2-large-xlsr-53-spanish`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish)
- Lenguaje: [`KenLM`](https://github.com/kpu/kenlm)
- Transcripción: [`pyctcdecode`](https://github.com/kensho-technologies/pyctcdecode)

---

## 📌 Nota

Este proyecto fue desarrollado para un Trabajo de Fin de Máster (TFM). Puede extenderse con fine-tuning, nuevos datasets, múltiples modelos y mejoras en el pipeline.

---
