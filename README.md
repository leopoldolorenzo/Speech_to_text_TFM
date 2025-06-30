Perfecto. Aquí tenés el `README.md` actualizado con:

* El **nombre oficial del proyecto**: `tfm_asr2025`
* Los **nombres completos de los tres autores**
* Todo el contenido técnico organizado y listo para dejarlo en el repositorio

---

### ✅ `README.md` final para `tfm_asr2025`

```markdown
# 🧠 tfm_asr2025 – Fine-tuning de Wav2Vec2 en Español con Common Voice y Modelo de Lenguaje

Este proyecto implementa un pipeline completo de **reconocimiento automático del habla (ASR)** en español mediante **fine-tuning del modelo Wav2Vec2.0** sobre el dataset **Common Voice**. Además, se incorpora un **modelo de lenguaje externo (KenLM)** para mejorar la decodificación usando `pyctcdecode`.

> Trabajo Fin de Máster – Máster en Inteligencia Artificial (UNIR)  
> Proyecto colaborativo de investigación aplicada sobre ASR en español.

---

## 👥 Autores

- **Leopoldo Lorenzo Fernández**
- **José Navacerrada Santiago**
- **Rafael Fernández Pedroche**

---

## 📁 Estructura del proyecto

```

TFM/
├── data/                  # Dataset, corpus LM, audios de prueba
├── models/                # Modelos entrenados y tokenizer
├── training/              # Checkpoints durante el entrenamiento
├── logs/                  # Logs de entrenamiento (TensorBoard)
├── scripts/               # Scripts para preprocesamiento, entrenamiento, evaluación
├── tools/                 # KenLM compilado y utilitarios
├── notebooks/             # Visualizaciones o pruebas exploratorias
├── requirements.txt       # Requisitos mínimos de Python
├── environment.yml        # Entorno Conda/Mamba completo
└── README.md              # Este archivo

````

---

## 🧪 Requisitos del entorno

### 🐍 Crear entorno con Mamba o Conda

```bash
mamba create -n asr_env python=3.10 -y
mamba activate asr_env
mamba install -c conda-forge --file requirements.txt
````

> Alternativamente:
> `conda env create -f environment.yml`

---

## 🚀 Pipeline de entrenamiento

1. **Preparar muestra de entrenamiento:**

```bash
python scripts/convertir_y_preparar_muestra_1.py
```

2. **Generar dataset compatible con HuggingFace:**

```bash
python scripts/generar_dataset_finetune_01.py
```

3. **Verificaciones previas:**

```bash
python scripts/verificar_dataset_entrenamiento.py
python scripts/verificar_cobertura_vocabulario.py
python scripts/verificar_tokenizer.py
```

4. **Entrenamiento principal:**

```bash
python scripts/entrenar_finetune_01.py
```

5. **Prueba de inferencia:**

```bash
python scripts/verificar_modelo_finetune.py
```

---

## 🧠 Modelo de Lenguaje (KenLM)

1. **Generar LM adaptado al vocabulario del modelo ASR:**

```bash
python scripts/generar_lm_finetune.py
```

Esto genera:

```
data/lm/modelo_finetune.arpa
data/lm/modelo_finetune.bin
```

2. **Uso con pyctcdecode para mejorar la transcripción.**

---

## 🔍 Evaluación comparativa

* **Sin modelo de lenguaje:**

```bash
python scripts/comparar_modelos_base_vs_finetune.py
```

* **Con modelo de lenguaje dedicado:**

```bash
python scripts/comparar_modelos_con_lm_dedicado.py
```

> El modelo base usa `modelo_limpio.bin`
> El modelo fine-tuneado usa `modelo_finetune.bin`

---

## 🔑 Licencia

Este proyecto es de uso **académico y educativo**.
Incluye únicamente datos públicos: Mozilla Common Voice y Hugging Face.

---

## 📦 Repositorio remoto

El repositorio se aloja de forma privada en NAS Synology:

```
ssh://leo@192.168.0.5:/volume2/Git/tfm_asr2025.git
```

```

---
