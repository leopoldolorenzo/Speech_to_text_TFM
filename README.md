# 🧠 tfm_asr2025 – Fine-tuning de Wav2Vec2 en Español con Common Voice y Modelo de Lenguaje

Este proyecto implementa un pipeline completo de **reconocimiento automático del habla (ASR)** en español mediante **fine-tuning del modelo Wav2Vec2.0** sobre el dataset **Common Voice**. Además, se incorpora un **modelo de lenguaje externo (KenLM)** para mejorar la decodificación usando `pyctcdecode`. El sistema se complementa con **diarización de hablantes (PyAnnote.audio)** y **traducción automática local (Helsinki-NLP)**.

> Trabajo Fin de Máster – Máster en Inteligencia Artificial (UNIR)  
> Proyecto colaborativo de investigación aplicada sobre ASR en español.

---

## 👥 Autores

- **Leopoldo Lorenzo Fernández**
- **José Fernando Navacerrada Santiago**
- **Rafael Fernández Pedroche**

---

## 📁 Estructura del proyecto

TFM/
├── data/ # Dataset, corpus LM, audios de prueba
├── models/ # Modelos entrenados y tokenizer
├── training/ # Checkpoints durante el entrenamiento
├── logs/ # Logs de entrenamiento (TensorBoard / W&B)
├── scripts/ # Scripts para preprocesamiento, entrenamiento, evaluación
├── tools/ # KenLM compilado, pyctcdecode y utilitarios
├── notebooks/ # Visualizaciones o pruebas exploratorias
├── frontend/ # Interfaz React para transcripción y visualización
├── backend/ # API FastAPI para inferencia y control del pipeline
├── requirements.txt # Requisitos mínimos de Python
├── environment.yml # Entorno Conda/Mamba completo
└── README.md # Este archivo

yaml
Copiar
Editar

---

## 🧪 Requisitos del entorno

### 🐍 Crear entorno con Mamba o Conda

```bash
mamba create -n asr_env python=3.10 -y
mamba activate asr_env
mamba install -c conda-forge --file requirements.txt
Alternativamente:
conda env create -f environment.yml

🚀 Pipeline de entrenamiento
1️⃣ Preparar muestra de entrenamiento:

bash
Copiar
Editar
python scripts/convertir_y_preparar_muestra_1.py
2️⃣ Generar dataset compatible con HuggingFace:

bash
Copiar
Editar
python scripts/generar_dataset_finetune_01.py
3️⃣ Verificaciones previas:

bash
Copiar
Editar
python scripts/verificar_dataset_entrenamiento.py
python scripts/verificar_cobertura_vocabulario.py
python scripts/verificar_tokenizer.py
4️⃣ Entrenamiento principal:

bash
Copiar
Editar
python scripts/entrenar_finetune_01.py
5️⃣ Prueba de inferencia:

bash
Copiar
Editar
python scripts/verificar_modelo_finetune.py
🧠 Modelo de Lenguaje (KenLM)
1️⃣ Generar LM adaptado al vocabulario del modelo ASR:

bash
Copiar
Editar
python scripts/generar_lm_finetune.py
Esto genera:

bash
Copiar
Editar
data/lm/modelo_finetune.arpa
data/lm/modelo_finetune.bin
2️⃣ Uso con pyctcdecode para mejorar la transcripción.

🗣️ Diarización e identificación de hablantes
El pipeline integra pyannote.audio para segmentación y etiquetado de intervenciones:

Detección de actividad de voz

Extracción de embeddings

Clustering jerárquico para agrupar hablantes

Scripts principales:

bash
Copiar
Editar
python scripts/diarizacion_pipeline.py
🌍 Traducción automática local
Se usa el modelo Helsinki-NLP/opus-mt-es-en:

bash
Copiar
Editar
python scripts/traducir_transcripcion.py
El modelo se descarga y guarda localmente en models/opus-mt-es-en/.

🔍 Evaluación comparativa
✅ Sin modelo de lenguaje:

bash
Copiar
Editar
python scripts/comparar_modelos_base_vs_finetune.py
✅ Con modelo de lenguaje dedicado:

bash
Copiar
Editar
python scripts/comparar_modelos_con_lm_dedicado.py
El modelo base usa modelo_limpio.bin
El modelo fine-tuneado usa modelo_finetune.bin

💻 Frontend + API
📌 API FastAPI:

bash
Copiar
Editar
uvicorn backend.main:app --reload
📌 Frontend React:

bash
Copiar
Editar
cd frontend
npm install
npm start
La interfaz permite:

Subir audios

Visualizar transcripciones con colores por hablante

Resaltado dinámico durante la reproducción

🔑 Licencia
Este proyecto es de uso académico y educativo.
Incluye únicamente datos públicos: Mozilla Common Voice, OpenSubtitles, CC100, Hugging Face.