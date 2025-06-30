"""
Script: generar_corpus_finetune_06.py

Objetivo:
  Limpiar y preparar el corpus textual original para que sea compatible con
  el vocabulario de 41 tokens del modelo ASR fine_tune_05.

Funciones:
  - Carga vocabulario JSON del tokenizer.
  - Normaliza y filtra el corpus original.
  - Guarda el corpus limpio listo para entrenamiento de LM.
"""

import json
import os

def limpiar_corpus(vocab_path: str, corpus_in: str, corpus_out: str) -> None:
    """
    Limpia el corpus original manteniendo solo caracteres válidos.

    Args:
        vocab_path (str): Ruta al vocabulario JSON del tokenizer.
        corpus_in (str): Archivo original con texto sin limpiar.
        corpus_out (str): Archivo destino para el corpus limpio.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"No se encontró vocabulario en {vocab_path}")
    if not os.path.exists(corpus_in):
        raise FileNotFoundError(f"No se encontró corpus original en {corpus_in}")

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    valid_chars = set()
    for token in vocab.keys():
        if token == " ":
            valid_chars.add("|")  # reemplazar espacio por pipe
        else:
            for c in token:
                valid_chars.add(c)

    print(f"Caracteres válidos para filtrar: {''.join(sorted(valid_chars))}")

    lineas_leidas = 0
    lineas_guardadas = 0

    with open(corpus_in, encoding="utf-8") as fin, open(corpus_out, "w", encoding="utf-8") as fout:
        for line in fin:
            lineas_leidas += 1
            line = line.strip().lower().replace(" ", "|")
            filtered = "".join(c for c in line if c in valid_chars)
            if filtered:
                fout.write(filtered + "\n")
                lineas_guardadas += 1

    print(f"Limpieza finalizada. Líneas leídas: {lineas_leidas}, guardadas: {lineas_guardadas}")
    print(f"Corpus limpio guardado en: {corpus_out}")

if __name__ == "__main__":
    vocab_path = "models/tokenizer_v2_base41/vocab.json"
    corpus_in = "data/lm/corpus.txt"
    corpus_out = "data/lm/corpus_finetune_06.txt"

    limpiar_corpus(vocab_path, corpus_in, corpus_out)
