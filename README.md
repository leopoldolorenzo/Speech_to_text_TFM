¡Perfecto! Te dejo a continuación un `README.md` completo, adaptado a tu proyecto de fine-tuning de modelos Wav2Vec2 con evaluación y uso de lenguaje externo (LM) en español. Incluye:

- Descripción del proyecto  
- Estructura del repositorio  
- Requisitos de entorno  
- Instrucciones de uso  
- Entrenamiento  
- Evaluación  
- Licencia  

---

### ✅ CONTENIDO PARA TU `README.md`

```markdown
# 🧠 Fine-tuning de Wav2Vec2 en Español con Common Voice y Language Model (TFM)

Este proyecto implementa un pipeline completo para el **fine-tuning** del modelo `wav2vec2-large-xlsr-53-spanish` usando el dataset **Common Voice** en español. Además, incorpora un **modelo de lenguaje externo (KenLM)** para mejorar la decodificación.

> Trabajo Fin de Máster – Reconocimiento Automático del Habla (ASR)

---

## 📁 Estructura del proyecto

```
TFM/
├── data/                  # Dataset, corpus LM, audios de prueba
├── models/                # Modelos entrenados y tokenizer
├── training/              # Checkpoints durante el entrenamiento
├── logs/                  # Logs de entrenamiento
├── scripts/               # Scripts para procesar, entrenar y evaluar
├── tools/                 # KenLM compilado y otros utilitarios
├── notebooks/             # (opcional) Visualizaciones o pruebas
├── requirements.txt       # Requisitos de Python
├── environment.yml        # Entorno Conda/Mamba
└── README.md              # Este archivo
```

---

## 🧪 Requisitos

### 🐍 Recomendado: crear entorno con Mamba o Conda

```bash
mamba create -n asr_env python=3.10 -y
mamba activate asr_env
mamba install -c conda-forge --file requirements.txt
```

> 💡 También podés usar: `conda env create -f environment.yml`

---

## 🚀 Entrenamiento paso a paso

1. **Convertir audios y preparar dataset:**

```bash
python scripts/convertir_y_preparar_muestra_1.py
```

2. **Generar dataset HF:**

```bash
python scripts/generar_dataset_finetune_01.py
```

3. **Verificar todo antes de entrenar:**

```bash
python scripts/verificar_dataset_entrenamiento.py
python scripts/verificar_cobertura_vocabulario.py
python scripts/verificar_tokenizer.py
```

4. **Lanzar entrenamiento:**

```bash
python scripts/entrenar_finetune_01.py
```

5. **Probar el modelo entrenado:**

```bash
python scripts/verificar_modelo_finetune.py
```

---

## 🧠 Modelo de Lenguaje (KenLM)

### Crear un LM adaptado al vocabulario del modelo fine-tuneado:

```bash
python scripts/generar_lm_finetune.py
```

Esto genera:

```
data/lm/modelo_finetune.arpa
data/lm/modelo_finetune.bin
```

---

## 🔍 Evaluación y comparación

### Comparar base vs fine-tune sin LM:

```bash
python scripts/comparar_modelos_base_vs_finetune.py
```

### Comparar ambos con su LM respectivo:

```bash
python scripts/comparar_modelos_con_lm_dedicado.py
```

> El modelo base usa `modelo_limpio.bin` y el fine-tune usa `modelo_finetune.bin`.

---

## 🔑 Licencia

Este proyecto es solo para **uso académico** y está basado en datos públicos (Common Voice, HuggingFace).

---

## ✍️ Autor

**Rafael FERNÁNDEZ PEDROCHE**
**José NAVACERRADA SANTIAGO**
**Leopoldo LORENZO FERNÁNDEZ**

Máster en Inteligencia Artificial  
UNIR
```

---

¿Querés que lo guarde como archivo directamente en tu proyecto? ¿O lo copias vos?